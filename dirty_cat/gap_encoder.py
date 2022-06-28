"""
Online Gamma-Poisson factorization of string arrays.
The principle is as follows:
    1. Given an input string array X, we build its bag-of-n-grams
       representation V (n_samples, vocab_size).
    2. Instead of using the n-grams counts as encodings, we look for low-
       dimensional representations by modeling n-grams counts as linear
       combinations of topics V = HW, with W (n_topics, vocab_size) the topics
       and H (n_samples, n_topics) the associated activations.
    3. Assuming that n-grams counts follow a Poisson law, we fit H and W to
       maximize the likelihood of the data, with a Gamma prior for the
       activations H to induce sparsity.
    4. In practice, this is equivalent to a non-negative matrix factorization
       with the Kullback-Leibler divergence as loss, and a Gamma prior on H.
       We thus optimize H and W with the multiplicative update method.
"""
import warnings
import numpy as np
from distutils.version import LooseVersion
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn.utils import check_random_state, gen_batches
from sklearn.utils.extmath import row_norms, safe_sparse_dot
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.fixes import _object_dtype_isnan
import pandas as pd
from .utils import check_input

if LooseVersion(sklearn_version) < LooseVersion('0.22'):
    from sklearn.cluster.k_means_ import _k_init
elif LooseVersion(sklearn_version) < LooseVersion('0.24'):
    from sklearn.cluster._kmeans import _k_init
else:
    from sklearn.cluster import kmeans_plusplus

if LooseVersion(sklearn_version) < LooseVersion('0.22'):
    from sklearn.decomposition.nmf import _beta_divergence
else:
    from sklearn.decomposition._nmf import _beta_divergence


class GapEncoderColumn(BaseEstimator, TransformerMixin):

    """See GapEncoder's docstring."""

    def __init__(self, n_components=10, batch_size=128, gamma_shape_prior=1.1,
                 gamma_scale_prior=1.0, rho=.95, rescale_rho=False,
                 hashing=False, hashing_n_features=2**12, init='k-means++',
                 tol=1e-4, min_iter=2, max_iter=5, ngram_range=(2, 4),
                 analyzer='char', add_words=False, random_state=None,
                 rescale_W=True, max_iter_e_step=20):

        self.ngram_range = ngram_range
        self.n_components = n_components
        self.gamma_shape_prior = gamma_shape_prior  # 'a' parameter
        self.gamma_scale_prior = gamma_scale_prior  # 'b' parameter
        self.rho = rho
        self.rho_ = self.rho
        self.rescale_rho = rescale_rho
        self.batch_size = batch_size
        self.tol = tol
        self.hashing = hashing
        self.hashing_n_features = hashing_n_features
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.init = init
        self.analyzer = analyzer
        self.add_words = add_words
        self.random_state = check_random_state(random_state)
        self.rescale_W = rescale_W
        self.max_iter_e_step = max_iter_e_step

    def _init_vars(self, X):
        """
        Build the bag-of-n-grams representation V of X and initialize
        the topics W.
        """
        # Init n-grams counts vectorizer
        if self.hashing:
            self.ngrams_count_ = HashingVectorizer(
                 analyzer=self.analyzer, ngram_range=self.ngram_range,
                 n_features=self.hashing_n_features,
                 norm=None, alternate_sign=False)
            if self.add_words: # Init a word counts vectorizer if needed
                self.word_count_ = HashingVectorizer(
                     analyzer='word',
                     n_features=self.hashing_n_features,
                     norm=None, alternate_sign=False)
        else:
            self.ngrams_count_ = CountVectorizer(
                 analyzer=self.analyzer, ngram_range=self.ngram_range,
                 dtype=np.float64)
            if self.add_words:
                self.word_count_ = CountVectorizer(dtype=np.float64)

        # Init H_dict_ with empty dict to train from scratch
        self.H_dict_ = dict()
        # Build the n-grams counts matrix unq_V on unique elements of X
        unq_X, lookup = np.unique(X, return_inverse=True)
        unq_V = self.ngrams_count_.fit_transform(unq_X)
        if self.add_words: # Add word counts to unq_V
            unq_V2 = self.word_count_.fit_transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format='csr')

        if not self.hashing: # Build n-grams/word vocabulary
            if LooseVersion(sklearn_version) < LooseVersion('1.0'):
                self.vocabulary = self.ngrams_count_.get_feature_names()
            else:
                self.vocabulary = self.ngrams_count_.get_feature_names_out()
            if self.add_words:
                if LooseVersion(sklearn_version) < LooseVersion('1.0'):
                    self.vocabulary = np.concatenate((
                        self.vocabulary,
                        self.word_count_.get_feature_names()
                    ))
                else:
                    self.vocabulary = np.concatenate((
                        self.vocabulary,
                        self.word_count_.get_feature_names_out()
                    ))
        _, self.n_vocab = unq_V.shape
        # Init the topics W given the n-grams counts V
        self.W_, self.A_, self.B_ = self._init_w(unq_V[lookup], X)
        # Init the activations unq_H of each unique input string
        unq_H = _rescale_h(unq_V, np.ones((len(unq_X), self.n_components)))
        # Update self.H_dict_ with unique input strings and their activations
        self.H_dict_.update(zip(unq_X, unq_H))
        if self.rescale_rho:
            # Make update rate per iteration independant of the batch_size
            self.rho_ = self.rho ** (self.batch_size / len(X))
        return unq_X, unq_V, lookup

    def _get_H(self, X):
        """
        Return the bag-of-n-grams representation of X.
        """
        H_out = np.empty((len(X), self.n_components))
        for x, h_out in zip(X, H_out):
            h_out[:] = self.H_dict_[x]
        return H_out

    def _init_w(self, V, X):
        """
        Initialize the topics W.
        If self.init='k-means++', we use the init method of
        sklearn.cluster.KMeans.
        If self.init='random', topics are initialized with a Gamma
        distribution.
        If self.init='k-means', topics are initialized with a KMeans on the
        n-grams counts.
        """
        if self.init == 'k-means++':
            if LooseVersion(sklearn_version) < LooseVersion('0.24'):
                W = _k_init(
                    V, self.n_components,
                    x_squared_norms=row_norms(V, squared=True),
                    random_state=self.random_state,
                    n_local_trials=None) + .1
            else:
                W, _ = kmeans_plusplus(
                    V, self.n_components,
                    x_squared_norms=row_norms(V, squared=True),
                    random_state=self.random_state,
                    n_local_trials=None)
                W = W + .1 # To avoid restricting topics to few n-grams only
        elif self.init == 'random':
            W = self.random_state.gamma(
                shape=self.gamma_shape_prior, scale=self.gamma_scale_prior,
                size=(self.n_components, self.n_vocab))
        elif self.init == 'k-means':
            prototypes = get_kmeans_prototypes(
                X, self.n_components, analyzer=self.analyzer, random_state=self.random_state)
            W = self.ngrams_count_.transform(prototypes).A + .1
            if self.add_words:
                W2 = self.word_count_.transform(prototypes).A + .1
                W = np.hstack((W, W2))
            # if k-means doesn't find the exact number of prototypes
            if W.shape[0] < self.n_components:
                if LooseVersion(sklearn_version) < LooseVersion('0.24'):
                    W2 = _k_init(
                        V, self.n_components - W.shape[0],
                        x_squared_norms=row_norms(V, squared=True),
                        random_state=self.random_state,
                        n_local_trials=None) + .1
                else:
                    W2, _ = kmeans_plusplus(
                        V, self.n_components - W.shape[0],
                        x_squared_norms=row_norms(V, squared=True),
                        random_state=self.random_state,
                        n_local_trials=None)
                    W2 = W2 + .1
                W = np.concatenate((W, W2), axis=0)
        else:
            raise AttributeError(
                'Initialization method %s does not exist.' % self.init)
        W /= W.sum(axis=1, keepdims=True)
        A = np.ones((self.n_components, self.n_vocab)) * 1e-10
        B = A.copy()
        return W, A, B

    def fit(self, X, y=None):
        """
        Fit the GapEncoder on batches of X.

        Parameters
        ----------
        X : array-like, shape (n_samples, )
            The string data to fit the model on.
        
        Returns
        -------
        self
        """
        # Check if first item has str or np.str_ type
        assert isinstance(X[0], str), "ERROR: Input data is not string."
        # Make n-grams counts matrix unq_V
        unq_X, unq_V, lookup = self._init_vars(X)
        n_batch = (len(X) - 1) // self.batch_size + 1
        del X
        # Get activations unq_H
        unq_H = self._get_H(unq_X)

        for n_iter_ in range(self.max_iter):
            # Loop over batches
            for i, (unq_idx, idx) in enumerate(batch_lookup(
              lookup, n=self.batch_size)):
                if i == n_batch-1:
                    W_last = self.W_.copy()
                # Update the activations unq_H
                unq_H[unq_idx] = _multiplicative_update_h(
                    unq_V[unq_idx], self.W_, unq_H[unq_idx],
                    epsilon=1e-3, max_iter=self.max_iter_e_step,
                    rescale_W=self.rescale_W,
                    gamma_shape_prior=self.gamma_shape_prior,
                    gamma_scale_prior=self.gamma_scale_prior)
                # Update the topics self.W_
                _multiplicative_update_w(
                    unq_V[idx], self.W_, self.A_, self.B_, unq_H[idx],
                    self.rescale_W, self.rho_)

                if i == n_batch-1:
                    # Compute the norm of the update of W in the last batch
                    W_change = np.linalg.norm(
                        self.W_ - W_last) / np.linalg.norm(W_last)

            if (W_change < self.tol) and (n_iter_ >= self.min_iter - 1):
                break # Stop if the change in W is smaller than the tolerance

        # Update self.H_dict_ with the learned encoded vectors (activations)
        self.H_dict_.update(zip(unq_X, unq_H))
        return self

    def get_feature_names(self, n_labels=3, prefix=''):
        """ Deprecated, use "get_feature_names_out"
        """
        warnings.warn(
            "get_feature_names is deprecated in scikit-learn > 1.0. "
            "use get_feature_names_out instead",
            DeprecationWarning,
            )
        return self.get_feature_names_out(n_labels=n_labels,
                                          prefix=prefix)

    def get_feature_names_out(self, n_labels=3, prefix=''):
        """
        Returns the labels that best summarize the learned components/topics.
        For each topic, labels with highest activations are selected.
        
        Parameters
        ----------
        
        n_labels : int, default=3
            The number of labels used to describe each topic.
        
        Returns
        -------
        
        topic_labels : list of strings
            The labels that best describe each topic.
        
        """
        vectorizer = CountVectorizer()
        vectorizer.fit(list(self.H_dict_.keys()))
        if LooseVersion(sklearn_version) < LooseVersion('1.0'):
            vocabulary = np.array(vectorizer.get_feature_names())
        else:
            vocabulary = np.array(vectorizer.get_feature_names_out())
        encoding = self.transform(np.array(vocabulary).reshape(-1))
        encoding = abs(encoding)
        encoding = encoding / np.sum(encoding, axis=1, keepdims=True)
        n_components = encoding.shape[1]
        topic_labels = []
        for i in range(n_components):
            x = encoding[:, i]
            labels = vocabulary[np.argsort(-x)[:n_labels]]
            topic_labels.append(labels)
        topic_labels = [prefix + ', '.join(label) for label in topic_labels]
        return topic_labels

    def score(self, X):
        """
        Returns the Kullback-Leibler divergence between the n-grams counts
        matrix V of X, and its non-negative factorization HW.

        Parameters
        ----------
        X : array-like (str), shape (n_samples, )
            The data to encode.

        Returns
        -------
        kl_divergence : float.
            The Kullback-Leibler divergence.
        """
        # Build n-grams/word counts matrix
        unq_X, lookup = np.unique(X, return_inverse=True)
        unq_V = self.ngrams_count_.transform(unq_X)
        if self.add_words:
            unq_V2 = self.word_count_.transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format='csr')

        self._add_unseen_keys_to_H_dict(unq_X)
        unq_H = self._get_H(unq_X)
        # Given the learnt topics W, optimize the activations H to fit V = HW
        for slice in gen_batches(n=unq_H.shape[0],
                                 batch_size=self.batch_size):
            unq_H[slice] = _multiplicative_update_h(
                unq_V[slice], self.W_, unq_H[slice],
                epsilon=1e-3, max_iter=self.max_iter_e_step,
                rescale_W=self.rescale_W,
                gamma_shape_prior=self.gamma_shape_prior,
                gamma_scale_prior=self.gamma_scale_prior)
        # Compute the KL divergence between V and HW
        kl_divergence = _beta_divergence(
            unq_V[lookup], unq_H[lookup], self.W_,
            'kullback-leibler', square_root=False)
        return kl_divergence

    def partial_fit(self, X, y=None):
        """
        Partial fit of the GapEncoder on X.
        To be used in a online learning procedure where batches of data are
        coming one by one.

        Parameters
        ----------
        X : array-like, shape (n_samples, )
            The string data to fit the model on.
        
        Returns
        -------
        self
        
        """
        
        # Init H_dict_ with empty dict if it's the first call of partial_fit
        if not hasattr(self, 'H_dict_'):
            self.H_dict_ = dict()
        # Check if first item has str or np.str_ type
        assert isinstance(X[0], str), "ERROR: Input data is not string."
        # Check if it is not the first batch
        if hasattr(self, 'vocabulary'): # Update unq_X, unq_V with new batch
            unq_X, lookup = np.unique(X, return_inverse=True)
            unq_V = self.ngrams_count_.transform(unq_X)
            if self.add_words:
                unq_V2 = self.word_count_.transform(unq_X)
                unq_V = sparse.hstack((unq_V, unq_V2), format='csr')

            unseen_X = np.setdiff1d(unq_X, np.array([*self.H_dict_]))
            unseen_V = self.ngrams_count_.transform(unseen_X)
            if self.add_words:
                unseen_V2 = self.word_count_.transform(unseen_X)
                unseen_V = sparse.hstack((unseen_V, unseen_V2), format='csr')

            if unseen_V.shape[0] != 0:
                unseen_H = _rescale_h(
                    unseen_V, np.ones((len(unseen_X), self.n_components)))
                for x, h in zip(unseen_X, unseen_H):
                    self.H_dict_[x] = h
                del unseen_H
            del unseen_X, unseen_V
        else: # If it is the first batch, call _init_vars to init unq_X, unq_V
            unq_X, unq_V, lookup = self._init_vars(X)

        unq_H = self._get_H(unq_X)
        # Update the activations unq_H
        unq_H = _multiplicative_update_h(
            unq_V, self.W_, unq_H,
            epsilon=1e-3, max_iter=self.max_iter_e_step,
            rescale_W=self.rescale_W,
            gamma_shape_prior=self.gamma_shape_prior,
            gamma_scale_prior=self.gamma_scale_prior)
        # Update the topics self.W_
        _multiplicative_update_w(
            unq_V[lookup], self.W_, self.A_, self.B_,
            unq_H[lookup], self.rescale_W, self.rho_)
        # Update self.H_dict_ with the learned encoded vectors (activations)
        self.H_dict_.update(zip(unq_X, unq_H))
        return self

    def _add_unseen_keys_to_H_dict(self, X):
        """
        Add activations of unseen string categories from X to H_dict.
        """
        unseen_X = np.setdiff1d(X, np.array([*self.H_dict_]))
        if unseen_X.size > 0:
            unseen_V = self.ngrams_count_.transform(unseen_X)
            if self.add_words:
                unseen_V2 = self.word_count_.transform(unseen_X)
                unseen_V = sparse.hstack((unseen_V, unseen_V2), format='csr')

            unseen_H = _rescale_h(
                unseen_V, np.ones((unseen_V.shape[0], self.n_components)))
            self.H_dict_.update(zip(unseen_X, unseen_H))

    def transform(self, X):
        """
        Return the encoded vectors (activations) H of input strings in X.
        Given the learnt topics W, the activations H are tuned to fit V = HW.

        Parameters
        ----------
        X : array-like, shape (n_samples)
            The string data to encode.

        Returns
        -------
        H : 2-d array, shape (n_samples, n_topics)
            Transformed input.
        """
        # Check if first item has str or np.str_ type
        assert isinstance(X[0], str), "ERROR: Input data is not string."
        unq_X = np.unique(X)
        # Build the n-grams counts matrix V for the string data to encode
        unq_V = self.ngrams_count_.transform(unq_X)
        if self.add_words: # Add words counts
            unq_V2 = self.word_count_.transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format='csr')
        # Add unseen strings in X to H_dict
        self._add_unseen_keys_to_H_dict(unq_X)
        unq_H = self._get_H(unq_X)
        # Loop over batches
        for slice in gen_batches(n=unq_H.shape[0],
                                 batch_size=self.batch_size):
            # Given the learnt topics W, optimize H to fit V = HW
            unq_H[slice] = _multiplicative_update_h(
                unq_V[slice], self.W_, unq_H[slice],
                epsilon=1e-3, max_iter=100,
                rescale_W=self.rescale_W,
                gamma_shape_prior=self.gamma_shape_prior,
                gamma_scale_prior=self.gamma_scale_prior)
        # Store and return the encoded vectors of X
        self.H_dict_.update(zip(unq_X, unq_H))
        return self._get_H(X)


class GapEncoder(BaseEstimator, TransformerMixin):
    """
    This encoder can be understood as a continuous encoding on a set of latent
    categories estimated from the data. The latent categories are built by
    capturing combinations of substrings that frequently co-occur.

    The GapEncoder supports online learning on batches of data for
    scalability through the partial_fit method.

    Parameters
    ----------

    n_components : int, default=10
        Number of latent categories used to model string data.

    batch_size : int, default=128
        Number of samples per batch.

    gamma_shape_prior : float, default=1.1
        Shape parameter for the Gamma prior distribution.

    gamma_scale_prior : float, default=1.0
        Scale parameter for the Gamma prior distribution.

    rho : float, default=0.95
        Weight parameter for the update of the W matrix.

    rescale_rho : bool, default=False
        If true, use rho ** (batch_size / len(X)) instead of rho to obtain an
        update rate per iteration that is independent of the batch size.

    hashing : bool, default=False
        If true, HashingVectorizer is used instead of CountVectorizer.
        It has the advantage of being very low memory scalable to large
        datasets as there is no need to store a vocabulary dictionary in
        memory.

    hashing_n_features : int, default=2**12
        Number of features for the HashingVectorizer. Only relevant if
        hashing=True.

    init : str, default='k-means++'
        Initialization method of the W matrix.
        Options: {'k-means++', 'random', 'k-means'}.
        If init='k-means++', we use the init method of sklearn.cluster.KMeans.
        If init='random', topics are initialized with a Gamma distribution.
        If init='k-means', topics are initialized with a KMeans on the n-grams
        counts. This usually makes convergence faster but is a bit slower.

    tol : float, default=1e-4
        Tolerance for the convergence of the matrix W.

    min_iter : int, default=2
        Minimum number of iterations on the input data.

    max_iter : int, default=5
        Maximum number of iterations on the input data.

    ngram_range : tuple, default=(2, 4)
        The range of ngram length that will be used to build the
        bag-of-n-grams representation of the input data.

    analyzer : str, default='char'.
        Analyzer parameter for the CountVectorizer/HashingVectorizer.
        Options: {‘word’, ‘char’, ‘char_wb’}, describing whether the matrix V
        to factorize should be made of word counts or character n-gram counts.
        Option ‘char_wb’ creates character n-grams only from text inside word
        boundaries; n-grams at the edges of words are padded with space.

    add_words : bool, default=False
        If true, add the words counts to the bag-of-n-grams representation
        of the input data.

    random_state : int or None, default=None
        Pass an int for reproducible output across multiple function calls.

    rescale_W : bool, default=True
        If true, the weight matrix W is rescaled at each iteration
        to have an l1 norm equal to 1 for each row.

    max_iter_e_step : int, default=20
        Maximum number of iterations to adjust the activations h at each step.

    handle_missing : 'error' or 'empty_impute' (default)
        Whether to raise an error or impute with empty string '' if missing
        values (NaN) are present during fit (default is to impute).
        In the inverse transform, the missing category will be denoted as None.


    Attributes
    ----------

    References
    ----------
    For a detailed description of the method, see
    `Encoding high-cardinality string categorical variables
    <https://hal.inria.fr/hal-02171256v4>`_ by Cerda, Varoquaux (2019).
    
    """

    def __init__(self, n_components=10, batch_size=128, gamma_shape_prior=1.1,
                 gamma_scale_prior=1.0, rho=.95, rescale_rho=False,
                 hashing=False, hashing_n_features=2**12, init='k-means++',
                 tol=1e-4, min_iter=2, max_iter=5, ngram_range=(2, 4),
                 analyzer='char', add_words=False, random_state=None,
                 rescale_W=True, max_iter_e_step=20, handle_missing='zero_impute'):

        self.ngram_range = ngram_range
        self.n_components = n_components
        self.gamma_shape_prior = gamma_shape_prior  # 'a' parameter
        self.gamma_scale_prior = gamma_scale_prior  # 'b' parameter
        self.rho = rho
        self.rho_ = self.rho
        self.rescale_rho = rescale_rho
        self.batch_size = batch_size
        self.tol = tol
        self.hashing = hashing
        self.hashing_n_features = hashing_n_features
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.init = init
        self.analyzer = analyzer
        self.add_words = add_words
        self.random_state = random_state
        self.rescale_W = rescale_W
        self.max_iter_e_step = max_iter_e_step
        self.handle_missing = handle_missing

    def _create_column_gap_encoder(self) -> GapEncoderColumn:
        return GapEncoderColumn(
            ngram_range=self.ngram_range,
            n_components=self.n_components,
            analyzer = self.analyzer,
            gamma_shape_prior=self.gamma_shape_prior,
            gamma_scale_prior=self.gamma_scale_prior,
            rho=self.rho,
            rescale_rho=self.rescale_rho,
            batch_size=self.batch_size,
            tol=self.tol,
            hashing=self.hashing,
            hashing_n_features=self.hashing_n_features,
            max_iter=self.max_iter,
            init=self.init,
            add_words=self.add_words,
            random_state=self.random_state,
            rescale_W=self.rescale_W,
            max_iter_e_step=self.max_iter_e_step,
        )

    def _handle_missing(self, X):
        """
        Imputes missing values with `` or raises an error
        Note: modifies the array in-place.
        """
        if self.handle_missing not in ['error', 'zero_impute']:
            raise ValueError(
                "handle_missing should be either 'error' or "
                f"'zero_impute', got {self.handle_missing!r}"
            )

        missing_mask = _object_dtype_isnan(X)

        if missing_mask.any():
            if self.handle_missing == 'error':
                raise ValueError('Input data contains missing values.')
            elif self.handle_missing == 'zero_impute':
                X[missing_mask] = ''

        return X
            
    def fit(self, X, y=None):
        """
        Fit the GapEncoder on batches of X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The string data to fit the model on.
        
        Returns
        -------
        self
        
        """
        # If X is a dataframe, store its column names
        if isinstance(X, pd.DataFrame):
            self.column_names_ = list(X.columns)
        # Check input data shape
        X = check_input(X)
        X = self._handle_missing(X)
        self.fitted_models_ = []
        for k in range(X.shape[1]):
            col_enc = self._create_column_gap_encoder()
            self.fitted_models_.append(col_enc.fit(X[:, k]))
        return self

    def transform(self, X):
        """
        Return the encoded vectors (activations) H of input strings in X.
        Given the learnt topics W, the activations H are tuned to fit V = HW.
        When X has several columns, they are encoded separately and
        then concatenated.
        
        Remark: calling transform mutliple times in a row on the same
        input X can give slightly different encodings. This is expected
        due to a caching mechanism to speed things up.        

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The string data to encode.

        Returns
        -------
        H : 2-d array, shape (n_samples, n_topics * n_features)
            Transformed input.
        """
        # Check input data shape
        X = check_input(X)
        X = self._handle_missing(X)
        X_enc = []
        for k in range(X.shape[1]):
            X_enc.append(self.fitted_models_[k].transform(X[:, k]))
        X_enc = np.hstack(X_enc)
        return X_enc

    def partial_fit(self, X, y=None):
        """
        Partial fit of the GapEncoder on X.
        To be used in a online learning procedure where batches of data are
        coming one by one.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The string data to fit the model on.
        
        Returns
        -------
        self
        
        """
        # If X is a dataframe, store its column names
        if isinstance(X, pd.DataFrame):
            self.column_names_ = list(X.columns)
        # Check input data shape
        X = check_input(X)
        X = self._handle_missing(X)
        # Init the `GapEncoderColumn` instances if the model was
        # not fitted already.
        if not hasattr(self, 'fitted_models_'):
            self.fitted_models_ = [
                self._create_column_gap_encoder() for _ in range(X.shape[1])
            ]
        for k in range(X.shape[1]):
            self.fitted_models_[k].partial_fit(X[:, k])
        return self

    def get_feature_names_out(self, col_names=None, n_labels=3):
        """
        Returns the labels that best summarize the learned components/topics.
        For each topic, labels with highest activations are selected.
        
        Parameters
        ----------
        
        col_names : {None, list or str}, default=None
            The column names to be added as prefixes before the labels.
            If col_names == None, no prefixes are used.
            If col_names == 'auto', column names are automatically defined:
                - if the input data was a dataframe, its column names are used
                - otherwise, 'col1', ..., 'colN' are used as prefixes
            Prefixes can be manually set by passing a list  for col_names.
            
        n_labels : int, default=3
            The number of labels used to describe each topic.
        
        Returns
        -------
        
        topic_labels : list of strings
            The labels that best describe each topic.
        
        """
        assert hasattr(self, 'fitted_models_'), (
            'ERROR: GapEncoder must be fitted first.')
        # Generate prefixes
        if isinstance(col_names, str) and col_names == 'auto':
            if hasattr(self, 'column_names_'): # Use column names
                prefixes = [s + ': ' for s in self.column_names_]
            else: # Use 'col1: ', ... 'colN: ' as prefixes
                prefixes = [f'col{k}: ' for k in range(len(self.fitted_models_))]
        elif col_names is None:  # Empty prefixes
            prefixes = [''] * len(self.fitted_models_)
        else:
            prefixes = [s + ': ' for s in col_names]
        labels = list()
        for k, enc in enumerate(self.fitted_models_):
            col_labels = enc.get_feature_names_out(n_labels, prefixes[k])
            labels.extend(col_labels)
        return labels
    
    def get_feature_names(
        self, input_features=None, col_names=None, n_labels=3
    ):
        """ Deprecated, use "get_feature_names_out"
        """
        warnings.warn(
            "get_feature_names is deprecated in scikit-learn > 1.0. "
            "use get_feature_names_out instead",
            DeprecationWarning,
            )
        return self.get_feature_names_out(col_names, n_labels)
        

    def score(self, X):
        """
        Returns the sum over the columns of X of the Kullback-Leibler
        divergence between the n-grams counts matrix V of X, and its
        non-negative factorization HW.

        Parameters
        ----------
        X : array-like (str), shape (n_samples, n_features)
            The data to encode.

        Returns
        -------
        kl_divergence : float.
            The Kullback-Leibler divergence.
        """
        X = check_input(X)
        kl_divergence = 0
        for k in range(X.shape[1]):
            kl_divergence += self.fitted_models_[k].score(X[:,k])
        return kl_divergence
        
def _rescale_W(W, A):
    """
    Rescale the topics W to have a L1-norm equal to 1.
    """
    s = W.sum(axis=1, keepdims=True)
    W /= s
    A /= s
    return


def _multiplicative_update_w(Vt, W, A, B, Ht, rescale_W, rho):
    """
    Multiplicative update step for the topics W.
    """
    A *= rho
    A += W * safe_sparse_dot(Ht.T, Vt.multiply(1 / (np.dot(Ht, W) + 1e-10)))
    B *= rho
    B += Ht.sum(axis=0).reshape(-1, 1)
    np.divide(A, B, out=W)
    if rescale_W:
        _rescale_W(W, A)
    return W, A, B


def _rescale_h(V, H):
    """
    Rescale the activations H.
    """
    epsilon = 1e-10  # in case of a document having length=0
    H *= np.maximum(epsilon, V.sum(axis=1).A)
    H /= H.sum(axis=1, keepdims=True)
    return H


def _multiplicative_update_h(Vt, W, Ht, epsilon=1e-3, max_iter=10,
                             rescale_W=False,
                             gamma_shape_prior=1.1, gamma_scale_prior=1.):
    """
    Multiplicative update step for the activations H.
    """
    if rescale_W:
        WT1 = 1 + 1 / gamma_scale_prior
        W_WT1 = W / WT1
    else:
        WT1 = np.sum(W, axis=1) + 1 / gamma_scale_prior
        W_WT1 = W / WT1.reshape(-1, 1)
    const = (gamma_shape_prior - 1) / WT1
    squared_epsilon = epsilon**2
    for vt, ht in zip(Vt, Ht):
        vt_ = vt.data
        idx = vt.indices
        W_WT1_ = W_WT1[:, idx]
        W_ = W[:, idx]
        squared_norm = 1
        for n_iter_ in range(max_iter):
            if squared_norm <= squared_epsilon:
                break
            aux = np.dot(W_WT1_, vt_ / (np.dot(ht, W_) + 1e-10))
            ht_out = ht * aux + const
            squared_norm = np.dot(
                ht_out - ht, ht_out - ht) / np.dot(ht, ht)
            ht[:] = ht_out
    return Ht


def batch_lookup(lookup, n=1):
    """ Make batches of the lookup array. """
    len_iter = len(lookup)
    for idx in range(0, len_iter, n):
        indices = lookup[slice(idx, min(idx + n, len_iter))]
        unq_indices = np.unique(indices)
        yield (unq_indices, indices)


def get_kmeans_prototypes(X, n_prototypes, analyzer='char', hashing_dim=128,
                          ngram_range=(2, 4), sparse=False,
                          sample_weight=None, random_state=None):
    """
    Computes prototypes based on:
      - dimensionality reduction (via hashing n-grams)
      - k-means clustering
      - nearest neighbor
    """
    vectorizer = HashingVectorizer(analyzer=analyzer, norm=None,
                                   alternate_sign=False,
                                   ngram_range=ngram_range,
                                   n_features=hashing_dim)
    projected = vectorizer.transform(X)
    if not sparse:
        projected = projected.toarray()
    kmeans = KMeans(n_clusters=n_prototypes, random_state=random_state)
    kmeans.fit(projected, sample_weight=sample_weight)
    centers = kmeans.cluster_centers_
    neighbors = NearestNeighbors()
    neighbors.fit(projected)
    indexes_prototypes = np.unique(neighbors.kneighbors(centers, 1)[-1])
    return np.sort(X[indexes_prototypes])
