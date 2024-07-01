"""
Implements the GapEncoder: a probabilistic encoder for categorical variables.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import scipy.sparse as sp
from scipy import sparse
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition._nmf import _beta_divergence
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state, gen_batches
from sklearn.utils.extmath import row_norms, safe_sparse_dot
from sklearn.utils.validation import _num_samples, check_is_fitted

from . import _dataframe as sbd
from ._on_each_column import RejectColumn, SingleColumnTransformer
from ._utils import unique_strings


class GapEncoder(TransformerMixin, SingleColumnTransformer):
    """Constructs latent topics with continuous encoding.

    This encoder can be understood as a continuous encoding on a set of latent
    categories estimated from the data. The latent categories are built by
    capturing combinations of substrings that frequently co-occur.

    The GapEncoder supports online learning on batches of
    data for scalability through the GapEncoder.partial_fit
    method.

    The principle is as follows:

    1. Given an input string array `X`, we build its bag-of-n-grams
       representation `V` (`n_samples`, `vocab_size`).
    2. Instead of using the n-grams counts as encodings, we look for low-
       dimensional representations by modeling n-grams counts as linear
       combinations of topics ``V = HW``, with `W` (`n_topics`, `vocab_size`)
       the topics and `H` (`n_samples`, `n_topics`) the associated activations.
    3. Assuming that n-grams counts follow a Poisson law, we fit `H` and `W` to
       maximize the likelihood of the data, with a Gamma prior for the
       activations `H` to induce sparsity.
    4. In practice, this is equivalent to a non-negative matrix factorization
       with the Kullback-Leibler divergence as loss, and a Gamma prior on `H`.
       We thus optimize `H` and `W` with the multiplicative update method.

    "Gap" stands for "Gamma-Poisson", the families of distributions that are
    used to model the importance of topics in a document (Gamma), and the term
    frequencies in a document (Poisson).

    Input columns that do not have a string or Categorical dtype are rejected
    by raising a ``RejectColumn`` exception.

    Parameters
    ----------
    n_components : int, optional, default=10
        Number of latent categories used to model string data.
    batch_size : int, optional, default=1024
        Number of samples per batch.
    gamma_shape_prior : float, optional, default=1.1
        Shape parameter for the Gamma prior distribution.
    gamma_scale_prior : float, optional, default=1.0
        Scale parameter for the Gamma prior distribution.
    rho : float, optional, default=0.95
        Weight parameter for the update of the `W` matrix.
    rescale_rho : bool, optional, default=False
        If `True`, use ``rho ** (batch_size / len(X))`` instead of rho to obtain
        an update rate per iteration that is independent of the batch size.
    hashing : bool, optional, default=False
        If `True`, HashingVectorizer is used instead of CountVectorizer.
        It has the advantage of being very low memory, scalable to large
        datasets as there is no need to store a vocabulary dictionary in
        memory.
    hashing_n_features : int, default=2**12
        Number of features for the HashingVectorizer.
        Only relevant if `hashing=True`.
    init : {'k-means++', 'random', 'k-means'}, default='k-means++'
        Initialization method of the `W` matrix.
        If `init='k-means++'`, we use the init method of KMeans.
        If `init='random'`, topics are initialized with a Gamma distribution.
        If `init='k-means'`, topics are initialized with a KMeans on the
        n-grams counts.
    max_iter : int, default=5
        Maximum number of iterations on the input data.
    ngram_range : int 2-tuple, default=(2, 4)
       The lower and upper boundaries of the range of n-values for different
        n-grams used in the string similarity. All values of `n` such
        that ``min_n <= n <= max_n`` will be used.
    analyzer : {'word', 'char', 'char_wb'}, default='char'
        Analyzer parameter for the HashingVectorizer / CountVectorizer.
        Describes whether the matrix `V` to factorize should be made of
        word counts or character-level n-gram counts.
        Option ‘char_wb’ creates character n-grams only from text inside word
        boundaries; n-grams at the edges of words are padded with space.
    add_words : bool, default=False
        If `True`, add the words counts to the bag-of-n-grams representation
        of the input data.
    random_state : int or RandomState, optional
        Random number generator seed for reproducible output across multiple
        function calls.
    rescale_W : bool, default=True
        If `True`, the weight matrix `W` is rescaled at each iteration
        to have a l1 norm equal to 1 for each row.
    max_iter_e_step : int, default=1
        Maximum number of iterations to adjust the activations h at each step.
    max_no_improvement : int, default=5
        Control early stopping based on the consecutive number of mini batches
        that do not yield an improvement on the smoothed cost function.
        To disable early stopping and run the process fully,
        set ``max_no_improvement=None``.
    verbose : int, default=0
        Verbosity level. The higher, the more granular the logging.

    Attributes
    ----------
    rho_ : float
        Effective update rate for the `W` matrix.
    fitted_models_ : list of GapEncoderColumn
        Column-wise fitted GapEncoders.
    column_names_ : list of str
        Column names of the data the Gap was fitted on.

    See Also
    --------
    MinHashEncoder :
        Encode string columns as a numeric array with the minhash method.
    SimilarityEncoder :
        Encode string columns as a numeric array with n-gram string similarity.
    deduplicate :
        Deduplicate data by hierarchically clustering similar strings.

    References
    ----------
    For a detailed description of the method, see
    `Encoding high-cardinality string categorical variables
    <https://hal.inria.fr/hal-02171256v4>`_ by Cerda, Varoquaux (2019).

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub import GapEncoder
    >>> enc = GapEncoder(n_components=2, random_state=0)

    Let's encode the following non-normalized data:

    >>> X = pd.Series(['Paris, FR', 'Paris', 'London, UK', 'Paris, France',
    ...                'london', 'London, England', 'London', 'Pqris'], name='city')
    >>> enc.fit(X)
    GapEncoder(n_components=2, random_state=0)

    The GapEncoder has found the following two topics:

    >>> enc.get_feature_names_out()
    ['city: england, london, uk', 'city: france, paris, pqris']

    It got it right, reccuring topics are "London" and "England" on the
    one side and "Paris" and "France" on the other.

    As this is a continuous encoding, we can look at the level of
    activation of each topic for each category:

    >>> enc.transform(X)
       city: england, london, uk  city: france, paris, pqris
    0                   0.051816                   10.548184
    1                   0.050134                    4.549866
    2                  12.046517                    0.053483
    3                   0.052270                   16.547730
    4                   6.049970                    0.050030
    5                  19.545227                    0.054773
    6                   6.049970                    0.050030
    7                   0.060120                    4.539880

    The higher the value, the bigger the correspondence with the topic.
    """

    def __init__(
        self,
        n_components=10,
        batch_size=1024,
        gamma_shape_prior=1.1,
        gamma_scale_prior=1.0,
        rho=0.95,
        rescale_rho=False,
        hashing=False,
        hashing_n_features=2**12,
        init="k-means++",
        max_iter=5,
        ngram_range=(2, 4),
        analyzer="char",
        add_words=False,
        random_state=None,
        rescale_W=True,
        max_iter_e_step=1,
        max_no_improvement=5,
        verbose=0,
    ):
        self.ngram_range = ngram_range
        self.n_components = n_components
        self.gamma_shape_prior = gamma_shape_prior  # 'a' parameter
        self.gamma_scale_prior = gamma_scale_prior  # 'b' parameter
        self.rho = rho
        self.rescale_rho = rescale_rho
        self.batch_size = batch_size
        self.hashing = hashing
        self.hashing_n_features = hashing_n_features
        self.max_iter = max_iter
        self.init = init
        self.analyzer = analyzer
        self.add_words = add_words
        self.random_state = random_state
        self.rescale_W = rescale_W
        self.max_iter_e_step = max_iter_e_step
        self.max_no_improvement = max_no_improvement
        self.verbose = verbose

    def _init_vars(self, X, is_null):
        """
        Build the bag-of-n-grams representation `V` of `X` and initialize
        the topics `W`.
        """
        # Init n-grams counts vectorizer
        if self.hashing:
            self.ngrams_count_ = HashingVectorizer(
                analyzer=self.analyzer,
                ngram_range=self.ngram_range,
                n_features=self.hashing_n_features,
                norm=None,
                alternate_sign=False,
            )
            if self.add_words:  # Init a word counts vectorizer if needed
                self.word_count_ = HashingVectorizer(
                    analyzer="word",
                    n_features=self.hashing_n_features,
                    norm=None,
                    alternate_sign=False,
                )
        else:
            self.ngrams_count_ = CountVectorizer(
                analyzer=self.analyzer,
                ngram_range=self.ngram_range,
                dtype=np.float64,
            )
            if self.add_words:
                self.word_count_ = CountVectorizer(dtype=np.float64)

        # Init H_dict_ with empty dict to train from scratch
        self.H_dict_ = dict()
        # Build the n-grams counts matrix unq_V on unique elements of X
        unq_X, lookup = unique_strings(X, is_null)
        unq_V = self.ngrams_count_.fit_transform(unq_X)
        if self.add_words:  # Add word counts to unq_V
            unq_V2 = self.word_count_.fit_transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format="csr")

        if not self.hashing:  # Build n-grams/word vocabulary
            self.vocabulary = self.ngrams_count_.get_feature_names_out()
            if self.add_words:
                self.vocabulary = np.concatenate(
                    (self.vocabulary, self.word_count_.get_feature_names_out())
                )
        _, self.n_vocab = unq_V.shape
        # Init the topics W given the n-grams counts V
        self.W_, self.A_, self.B_ = self._init_w(unq_V[lookup], X)
        # Init the activations unq_H of each unique input string
        unq_H = _rescale_h(unq_V, np.ones((len(unq_X), self.n_components)))
        # Update self.H_dict_ with unique input strings and their activations
        self.H_dict_.update(zip(unq_X, unq_H))
        if self.rescale_rho:
            # Make update rate per iteration independent of the batch_size
            self.rho_ = self.rho ** (self.batch_size / len(X))
        return unq_X, unq_V, lookup

    def _get_H(self, X):
        """
        Return the bag-of-n-grams representation of `X`.
        """
        H_out = np.empty((len(X), self.n_components))
        for x, h_out in zip(X, H_out):
            if not isinstance(x, str):
                x = ""
            h_out[:] = self.H_dict_[x]
        return H_out

    def _init_w(self, V, X):
        """
        Initialize the topics `W`.
        If `self.init='k-means++'`, we use the init method of
        sklearn.cluster.KMeans.
        If `self.init='random'`, topics are initialized with a Gamma
        distribution.
        If `self.init='k-means'`, topics are initialized with a KMeans on the
        n-grams counts.
        """
        if self.init == "k-means++":
            W, _ = kmeans_plusplus(
                V,
                self.n_components,
                x_squared_norms=row_norms(V, squared=True),
                random_state=self._random_state,
                n_local_trials=None,
            )
            W = W + 0.1  # To avoid restricting topics to a few n-grams only
        elif self.init == "random":
            W = self._random_state.gamma(
                shape=self.gamma_shape_prior,
                scale=self.gamma_scale_prior,
                size=(self.n_components, self.n_vocab),
            )
        elif self.init == "k-means":
            prototypes = get_kmeans_prototypes(
                X,
                self.n_components,
                analyzer=self.analyzer,
                random_state=self._random_state,
            )
            W = self.ngrams_count_.transform(prototypes).toarray() + 0.1
            if self.add_words:
                W2 = self.word_count_.transform(prototypes).toarray() + 0.1
                W = np.hstack((W, W2))
            # if k-means doesn't find the exact number of prototypes
            if W.shape[0] < self.n_components:
                W2, _ = kmeans_plusplus(
                    V,
                    self.n_components - W.shape[0],
                    x_squared_norms=row_norms(V, squared=True),
                    random_state=self._random_state,
                    n_local_trials=None,
                )
                W2 = W2 + 0.1
                W = np.concatenate((W, W2), axis=0)
        else:
            raise ValueError(f"Initialization method {self.init!r} does not exist. ")
        W /= W.sum(axis=1, keepdims=True)
        A = np.ones((self.n_components, self.n_vocab)) * 1e-10
        B = A.copy()
        return W, A, B

    def _minibatch_convergence(
        self,
        batch_size,
        batch_cost,
        n_samples,
        step,
        n_steps,
    ):
        """
        Helper function to encapsulate the early stopping logic.

        Parameters
        ----------
        batch_size : int
            The size of the current batch.
        batch_cost : float
            The cost (KL score) of the current batch.
        n_samples : int
            The total number of samples in X.
        step : int
            The current step (for verbose mode).
        n_steps : int
            The total number of steps (for verbose mode).

        Returns
        -------
        bool
            Whether the algorithm should stop or not.
        """
        # adapted from sklearn.decomposition.MiniBatchNMF

        # counts steps starting from 1 for user friendly verbose mode.
        step = step + 1

        # Ignore first iteration because H is not updated yet.
        if step == 1:
            if self.verbose:
                print(f"Minibatch step {step}/{n_steps}: mean batch cost: {batch_cost}")
            return False

        # Compute an Exponentially Weighted Average of the cost function to
        # monitor the convergence while discarding minibatch-local stochastic
        # variability: https://en.wikipedia.org/wiki/Moving_average
        if self._ewa_cost is None:
            self._ewa_cost = batch_cost
        else:
            alpha = batch_size / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_cost = self._ewa_cost * (1 - alpha) + batch_cost * alpha

        # Log progress to be able to monitor convergence
        if self.verbose:
            print(
                f"Minibatch step {step}/{n_steps}: mean batch cost: "
                f"{batch_cost}, ewa cost: {self._ewa_cost}"
            )

        # Early stopping heuristic due to lack of improvement on smoothed
        # cost function
        if self._ewa_cost_min is None or self._ewa_cost < self._ewa_cost_min:
            self._no_improvement = 0
            self._ewa_cost_min = self._ewa_cost
        else:
            self._no_improvement += 1

        if (
            self.max_no_improvement is not None
            and self._no_improvement >= self.max_no_improvement
        ):
            if self.verbose:
                print(
                    "Converged (lack of improvement in objective function) "
                    f"at step {step}/{n_steps}"
                )
            return True

        return False

    def _check_analyzer(self):
        if not isinstance(self.analyzer, str) or self.analyzer not in (
            "word",
            "char",
            "char_wb",
        ):
            raise ValueError("analyzer should be one of ['word', 'char', 'char_wb'].")

    def fit(self, X, y=None):
        """
        Fit the GapEncoder on `X`.

        Parameters
        ----------
        X : Column, shape (n_samples, )
            The string data to fit the model on.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        GapEncoderColumn
            The fitted GapEncoderColumn instance (self).
        """
        self._check_analyzer()
        self._check_input_type(X)
        # Check that n_samples >= n_components
        n_samples = _num_samples(X)
        if n_samples < self.n_components:
            raise ValueError(
                f"n_samples={n_samples} should be >= n_components={self.n_components}. "
            )
        self._input_name = sbd.name(X)
        self._random_state = check_random_state(self.random_state)
        is_null = sbd.to_numpy(sbd.is_null(X))
        X = sbd.to_numpy(X)
        # Copy parameter rho
        self.rho_ = self.rho
        # Attributes to monitor the convergence
        self._ewa_cost = None
        self._ewa_cost_min = None
        self._no_improvement = 0
        # Make n-grams counts matrix unq_V
        unq_X, unq_V, lookup = self._init_vars(X, is_null)
        n_batch = (len(X) - 1) // self.batch_size + 1
        n_samples = len(X)
        del X
        # Get activations unq_H
        unq_H = self._get_H(unq_X)
        converged = False
        for n_iter_ in range(self.max_iter):
            # Loop over batches
            for i, (unq_idx, idx) in enumerate(batch_lookup(lookup, n=self.batch_size)):
                # Update activations unq_H
                unq_H[unq_idx] = _multiplicative_update_h(
                    unq_V[unq_idx],
                    self.W_,
                    unq_H[unq_idx],
                    epsilon=1e-3,
                    max_iter=self.max_iter_e_step,
                    rescale_W=self.rescale_W,
                    gamma_shape_prior=self.gamma_shape_prior,
                    gamma_scale_prior=self.gamma_scale_prior,
                )
                # Update the topics self.W_
                _multiplicative_update_w(
                    unq_V[idx],
                    self.W_,
                    self.A_,
                    self.B_,
                    unq_H[idx],
                    self.rescale_W,
                    self.rho_,
                )
                batch_cost = _beta_divergence(
                    unq_V[idx],
                    unq_H[idx],
                    self.W_,
                    "kullback-leibler",
                    square_root=False,
                ) / len(idx)
                if self._minibatch_convergence(
                    batch_size=len(idx),
                    batch_cost=batch_cost,
                    n_samples=n_samples,
                    step=n_iter_ * n_batch + i,
                    n_steps=self.max_iter * n_batch,
                ):
                    converged = True
                    break
            if converged:
                break

        # Update self.H_dict_ with the learned encoded vectors (activations)
        self.H_dict_.update(zip(unq_X, unq_H))
        return self

    def get_feature_names_out(self, n_labels=3):
        """
        Return the labels that best summarize the learned components/topics.

        For each topic, labels with the highest activations are selected.

        Parameters
        ----------
        n_labels : int, default=3
            The number of labels used to describe each topic.

        Returns
        -------
        list of str
            The labels that best describe each topic.
        """
        check_is_fitted(self, "H_dict_")
        vectorizer = CountVectorizer()
        try:
            vectorizer.fit(list(self.H_dict_.keys()))
        except ValueError:
            # The vectorizer failed to find words, we need to switch to
            # char-level representation
            vectorizer = CountVectorizer(analyzer="char_wb")
            vectorizer.fit(list(self.H_dict_.keys()))
        vocabulary = np.array(vectorizer.get_feature_names_out())
        encoding = self._transform(vocabulary, np.zeros(len(vocabulary), dtype="bool"))
        encoding = abs(encoding)
        encoding = encoding / np.sum(encoding, axis=1, keepdims=True)
        n_components = encoding.shape[1]
        topic_labels = []
        for i in range(n_components):
            x = encoding[:, i]
            labels = vocabulary[np.argsort(-x)[:n_labels]]
            label = ", ".join(labels)
            if self._input_name:
                label = f"{self._input_name}: {label}"
            # Avoid having the same name twice for the different features
            if label in topic_labels:
                label = f"{label} ({i})"
            topic_labels.append(label)
        return topic_labels

    def score(self, X):
        """Score this instance of `X`.

        Returns the Kullback-Leibler divergence between the n-grams counts
        matrix `V` of `X`, and its non-negative factorization `HW`.

        Parameters
        ----------
        X : Column, shape (n_samples, )
            The data to encode.

        Returns
        -------
        float
            The Kullback-Leibler divergence.
        """
        is_null = sbd.to_numpy(sbd.is_null(X))
        X = sbd.to_numpy(X)
        # Build n-grams/word counts matrix
        unq_X, lookup = unique_strings(X, is_null)
        unq_V = self.ngrams_count_.transform(unq_X)
        if self.add_words:
            unq_V2 = self.word_count_.transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format="csr")

        self._add_unseen_keys_to_H_dict(unq_X)
        unq_H = self._get_H(unq_X)
        # Given the learnt topics W, optimize the activations H to fit V = HW
        for slice in gen_batches(n=unq_H.shape[0], batch_size=self.batch_size):
            unq_H[slice] = _multiplicative_update_h(
                unq_V[slice],
                self.W_,
                unq_H[slice],
                epsilon=1e-3,
                max_iter=self.max_iter_e_step,
                rescale_W=self.rescale_W,
                gamma_shape_prior=self.gamma_shape_prior,
                gamma_scale_prior=self.gamma_scale_prior,
            )
        # Compute the KL divergence between V and HW
        kl_divergence = _beta_divergence(
            unq_V[lookup], unq_H[lookup], self.W_, "kullback-leibler", square_root=False
        )
        return kl_divergence

    def partial_fit(self, X, y=None):
        """Partial fit this instance on `X`.

        To be used in an online learning procedure where batches of data are
        coming one by one.

        Parameters
        ----------
        X : Column, shape (n_samples, )
            The string data to fit the model on.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        GapEncoderColumn
            The fitted GapEncoderColumn instance (self).
        """
        self._check_analyzer()
        self._check_input_type(X)
        if not hasattr(self, "_input_name"):
            self._input_name = sbd.name(X)
        if not hasattr(self, "_random_state"):
            self._random_state = check_random_state(self.random_state)
        is_null = sbd.to_numpy(sbd.is_null(X))
        X = sbd.to_numpy(X)
        # Init H_dict_ with empty dict if it's the first call of partial_fit
        if not hasattr(self, "H_dict_"):
            self.H_dict_ = dict()
        # Same thing for the rho_ parameter
        if not hasattr(self, "rho_"):
            self.rho_ = self.rho
        # Check if it is not the first batch
        if hasattr(self, "vocabulary"):  # Update unq_X, unq_V with new batch
            unq_X, lookup = unique_strings(X, is_null)
            unq_V = self.ngrams_count_.transform(unq_X)
            if self.add_words:
                unq_V2 = self.word_count_.transform(unq_X)
                unq_V = sparse.hstack((unq_V, unq_V2), format="csr")

            unseen_X = np.setdiff1d(unq_X, np.array([*self.H_dict_]))
            unseen_V = self.ngrams_count_.transform(unseen_X)
            if self.add_words:
                unseen_V2 = self.word_count_.transform(unseen_X)
                unseen_V = sparse.hstack((unseen_V, unseen_V2), format="csr")

            if unseen_V.shape[0] != 0:
                unseen_H = _rescale_h(
                    unseen_V, np.ones((len(unseen_X), self.n_components))
                )
                for x, h in zip(unseen_X, unseen_H):
                    self.H_dict_[x] = h
                del unseen_H
            del unseen_X, unseen_V
        else:  # If it is the first batch, call _init_vars to init unq_X, unq_V
            unq_X, unq_V, lookup = self._init_vars(X, is_null)

        unq_H = self._get_H(unq_X)
        # Update unq_H, the activations
        unq_H = _multiplicative_update_h(
            unq_V,
            self.W_,
            unq_H,
            epsilon=1e-3,
            max_iter=self.max_iter_e_step,
            rescale_W=self.rescale_W,
            gamma_shape_prior=self.gamma_shape_prior,
            gamma_scale_prior=self.gamma_scale_prior,
        )
        # Update the topics self.W_
        _multiplicative_update_w(
            unq_V[lookup],
            self.W_,
            self.A_,
            self.B_,
            unq_H[lookup],
            self.rescale_W,
            self.rho_,
        )
        # Update self.H_dict_ with the learned encoded vectors (activations)
        self.H_dict_.update(zip(unq_X, unq_H))
        return self

    def _add_unseen_keys_to_H_dict(self, X):
        """
        Add activations of unseen string categories from `X` to `H_dict`.
        """
        unseen_X = np.setdiff1d(X, np.array([*self.H_dict_]))
        if unseen_X.size > 0:
            unseen_V = self.ngrams_count_.transform(unseen_X)
            if self.add_words:
                unseen_V2 = self.word_count_.transform(unseen_X)
                unseen_V = sparse.hstack((unseen_V, unseen_V2), format="csr")

            unseen_H = _rescale_h(
                unseen_V, np.ones((unseen_V.shape[0], self.n_components))
            )
            self.H_dict_.update(zip(unseen_X, unseen_H))

    def transform(self, X):
        """Return the encoded vectors (activations) `H` of input strings in `X`.

        Parameters
        ----------
        X : Column, shape (n_samples)
            The string data to encode.

        Returns
        -------
        DataFrame, shape (n_samples, n_topics)
            Transformed input.
        """
        check_is_fitted(self, "H_dict_")
        # rejecting columns is only for fit, so we raise a plain ValueError here
        self._check_input_type(X, err_type=ValueError)
        is_null = sbd.to_numpy(sbd.is_null(X))
        result = self._transform(sbd.to_numpy(X), is_null)
        names = self.get_feature_names_out()
        result = sbd.make_dataframe_like(X, dict(zip(names, result.T)))
        result = sbd.copy_index(X, result)
        return result

    def _check_input_type(self, X, err_type=RejectColumn):
        if sbd.is_categorical(X) or sbd.is_string(X):
            return
        raise err_type(f"Column {sbd.name(X)!r} does not contain strings.")

    def _transform(self, X, is_null):
        # Copy the state of H before continuing fitting it
        pre_trans_H_dict_ = deepcopy(self.H_dict_)
        unq_X, _ = unique_strings(X, is_null)
        # Build the n-grams counts matrix V for the string data to encode
        unq_V = self.ngrams_count_.transform(unq_X)
        if self.add_words:  # Add words counts
            unq_V2 = self.word_count_.transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format="csr")
        # Add unseen strings in X to H_dict
        self._add_unseen_keys_to_H_dict(unq_X)
        unq_H = self._get_H(unq_X)
        # Loop over batches
        for slc in gen_batches(n=unq_H.shape[0], batch_size=self.batch_size):
            # Given the learnt topics W, optimize H to fit V = HW
            unq_H[slc] = _multiplicative_update_h(
                unq_V[slc],
                self.W_,
                unq_H[slc],
                epsilon=1e-3,
                max_iter=100,
                rescale_W=self.rescale_W,
                gamma_shape_prior=self.gamma_shape_prior,
                gamma_scale_prior=self.gamma_scale_prior,
            )
        # Store and return the encoded vectors of X
        self.H_dict_.update(zip(unq_X, unq_H))
        result = self._get_H(X)
        # Restore H
        self.H_dict_ = pre_trans_H_dict_
        return result


def _rescale_W(W, A):
    """
    Rescale the topics `W` to have a L1-norm equal to 1.
    Note that they are modified in-place.
    """
    s = W.sum(axis=1, keepdims=True)
    W /= s
    A /= s


def _special_sparse_dot(H, W, X):
    """Computes np.dot(H, W), only where X is non zero."""
    # adapted from sklearn.decomposition.MiniBatchNMF
    if sp.issparse(X):
        ii, jj = X.nonzero()
        n_vals = ii.shape[0]
        dot_vals = np.empty(n_vals)
        n_components = H.shape[1]
        batch_size = max(n_components, n_vals // n_components)
        for start in range(0, n_vals, batch_size):
            batch = slice(start, start + batch_size)
            dot_vals[batch] = np.multiply(H[ii[batch], :], W.T[jj[batch], :]).sum(
                axis=1
            )

        HW = sp.coo_matrix((dot_vals, (ii, jj)), shape=X.shape)
        # in sklearn, it was return WH.tocsr(), but it breaks the code in our case
        # I'm not sure why
        return HW
    else:
        return np.dot(H, W)


def _multiplicative_update_w(
    Vt,
    W,
    A,
    B,
    Ht,
    rescale_W,
    rho,
):
    """
    Multiplicative update step for the topics `W`.
    """
    A *= rho
    HtW = _special_sparse_dot(Ht, W, Vt)
    Vt_data = Vt.data
    HtW_data = HtW.data
    np.divide(Vt_data, HtW_data + 1e-10, out=Vt_data)
    HtVt = safe_sparse_dot(Ht.T, Vt)
    A += W * HtVt
    B *= rho
    B += Ht.sum(axis=0).reshape(-1, 1)
    np.divide(A, B, out=W)
    if rescale_W:
        _rescale_W(W, A)
    return W, A, B


def _rescale_h(V, H):
    """
    Rescale the activations `H`.
    """
    epsilon = 1e-10  # in case of a document having length=0
    H *= np.maximum(epsilon, V.sum(axis=1).A)
    H /= H.sum(axis=1, keepdims=True)
    return H


def _multiplicative_update_h(
    Vt,
    W,
    Ht,
    epsilon=1e-3,
    max_iter=10,
    rescale_W=False,
    gamma_shape_prior=1.1,
    gamma_scale_prior=1.0,
):
    """
    Multiplicative update step for the activations `H`.
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
        for _ in range(max_iter):
            if squared_norm <= squared_epsilon:
                break
            aux = np.dot(W_WT1_, vt_ / (np.dot(ht, W_) + 1e-10))
            ht_out = ht * aux + const
            squared_norm = np.dot(ht_out - ht, ht_out - ht) / np.dot(ht, ht)
            ht[:] = ht_out
    return Ht


def batch_lookup(
    lookup,
    n=1,
):
    """
    Make batches of the lookup array.
    """
    len_iter = len(lookup)
    for idx in range(0, len_iter, n):
        indices = lookup[slice(idx, min(idx + n, len_iter))]
        unq_indices = np.unique(indices)
        yield unq_indices, indices


def get_kmeans_prototypes(
    X,
    n_prototypes,
    analyzer="char",
    hashing_dim=128,
    ngram_range=(2, 4),
    sparse=False,
    sample_weight=None,
    random_state=None,
):
    """
    Computes prototypes based on:
      - dimensionality reduction (via hashing n-grams)
      - k-means clustering
      - nearest neighbor
    """
    vectorizer = HashingVectorizer(
        analyzer=analyzer,
        norm=None,
        alternate_sign=False,
        ngram_range=ngram_range,
        n_features=hashing_dim,
    )
    projected = vectorizer.transform(X)
    if not sparse:
        projected = projected.toarray()
    kmeans = KMeans(n_clusters=n_prototypes, n_init=10, random_state=random_state)
    kmeans.fit(projected, sample_weight=sample_weight)
    centers = kmeans.cluster_centers_
    neighbors = NearestNeighbors()
    neighbors.fit(projected)
    indexes_prototypes = np.unique(neighbors.kneighbors(centers, 1)[-1])
    return np.sort(X[indexes_prototypes])
