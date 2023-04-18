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
from copy import deepcopy
from typing import Dict, Generator, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import RandomState
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state, gen_batches
from sklearn.utils.extmath import row_norms, safe_sparse_dot
from sklearn.utils.fixes import _object_dtype_isnan
from sklearn.utils.validation import check_is_fitted

from ._utils import check_input, parse_version

if parse_version(sklearn_version) < parse_version("0.24"):
    from sklearn.cluster._kmeans import _k_init
else:
    from sklearn.cluster import kmeans_plusplus

from sklearn.decomposition._nmf import _beta_divergence

# Ignore lines too long, as some things in the docstring cannot be cut.
# flake8: noqa: E501


class GapEncoderColumn(BaseEstimator, TransformerMixin):

    """See GapEncoder's docstring."""

    rho_: float
    H_dict_: Dict[np.ndarray, np.ndarray]

    def __init__(
        self,
        n_components: int = 10,
        batch_size: int = 128,
        gamma_shape_prior: float = 1.1,
        gamma_scale_prior: float = 1.0,
        rho: float = 0.95,
        rescale_rho: bool = False,
        hashing: bool = False,
        hashing_n_features: int = 2**12,
        init: Literal["k-means++", "random", "k-means"] = "k-means++",
        tol: float = 1e-4,
        min_iter: int = 2,
        max_iter: int = 5,
        ngram_range: Tuple[int, int] = (2, 4),
        analyzer: Literal["word", "char", "char_wb"] = "char",
        add_words: bool = False,
        random_state: Optional[Union[int, RandomState]] = None,
        rescale_W: bool = True,
        max_iter_e_step: int = 20,
    ):
        self.ngram_range = ngram_range
        self.n_components = n_components
        self.gamma_shape_prior = gamma_shape_prior  # 'a' parameter
        self.gamma_scale_prior = gamma_scale_prior  # 'b' parameter
        self.rho = rho
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

    def _init_vars(self, X) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the bag-of-n-grams representation V of X and initialize
        the topics W.
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
                analyzer=self.analyzer, ngram_range=self.ngram_range, dtype=np.float64
            )
            if self.add_words:
                self.word_count_ = CountVectorizer(dtype=np.float64)

        # Init H_dict_ with empty dict to train from scratch
        self.H_dict_ = dict()
        # Build the n-grams counts matrix unq_V on unique elements of X
        unq_X, lookup = np.unique(X, return_inverse=True)
        unq_V = self.ngrams_count_.fit_transform(unq_X)
        if self.add_words:  # Add word counts to unq_V
            unq_V2 = self.word_count_.fit_transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format="csr")

        if not self.hashing:  # Build n-grams/word vocabulary
            if parse_version(sklearn_version) < parse_version("1.0"):
                self.vocabulary = self.ngrams_count_.get_feature_names()
            else:
                self.vocabulary = self.ngrams_count_.get_feature_names_out()
            if self.add_words:
                if parse_version(sklearn_version) < parse_version("1.0"):
                    self.vocabulary = np.concatenate(
                        (self.vocabulary, self.word_count_.get_feature_names())
                    )
                else:
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

    def _get_H(self, X: np.array) -> np.array:
        """
        Return the bag-of-n-grams representation of X.
        """
        H_out = np.empty((len(X), self.n_components))
        for x, h_out in zip(X, H_out):
            h_out[:] = self.H_dict_[x]
        return H_out

    def _init_w(self, V: np.array, X) -> Tuple[np.array, np.array, np.array]:
        """
        Initialize the topics W.
        If self.init='k-means++', we use the init method of
        sklearn.cluster.KMeans.
        If self.init='random', topics are initialized with a Gamma
        distribution.
        If self.init='k-means', topics are initialized with a KMeans on the
        n-grams counts.
        """
        if self.init == "k-means++":
            if parse_version(sklearn_version) < parse_version("0.24"):
                W = (
                    _k_init(
                        V,
                        self.n_components,
                        x_squared_norms=row_norms(V, squared=True),
                        random_state=self.random_state,
                        n_local_trials=None,
                    )
                    + 0.1
                )
            else:
                W, _ = kmeans_plusplus(
                    V,
                    self.n_components,
                    x_squared_norms=row_norms(V, squared=True),
                    random_state=self.random_state,
                    n_local_trials=None,
                )
                W = W + 0.1  # To avoid restricting topics to a few n-grams only
        elif self.init == "random":
            W = self.random_state.gamma(
                shape=self.gamma_shape_prior,
                scale=self.gamma_scale_prior,
                size=(self.n_components, self.n_vocab),
            )
        elif self.init == "k-means":
            prototypes = get_kmeans_prototypes(
                X,
                self.n_components,
                analyzer=self.analyzer,
                random_state=self.random_state,
            )
            W = self.ngrams_count_.transform(prototypes).A + 0.1
            if self.add_words:
                W2 = self.word_count_.transform(prototypes).A + 0.1
                W = np.hstack((W, W2))
            # if k-means doesn't find the exact number of prototypes
            if W.shape[0] < self.n_components:
                if parse_version(sklearn_version) < parse_version("0.24"):
                    W2 = (
                        _k_init(
                            V,
                            self.n_components - W.shape[0],
                            x_squared_norms=row_norms(V, squared=True),
                            random_state=self.random_state,
                            n_local_trials=None,
                        )
                        + 0.1
                    )
                else:
                    W2, _ = kmeans_plusplus(
                        V,
                        self.n_components - W.shape[0],
                        x_squared_norms=row_norms(V, squared=True),
                        random_state=self.random_state,
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

    def fit(self, X, y=None) -> "GapEncoderColumn":
        """
        Fit the GapEncoder on batches of X.

        Parameters
        ----------
        X : array-like, shape (n_samples, )
            The string data to fit the model on.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        self
            Fitting GapEncoderColumn instance.
        """
        # Copy parameter rho
        self.rho_ = self.rho
        # Check if first item has str or np.str_ type
        assert isinstance(X[0], str), "Input data is not string. "
        # Make n-grams counts matrix unq_V
        unq_X, unq_V, lookup = self._init_vars(X)
        n_batch = (len(X) - 1) // self.batch_size + 1
        del X
        # Get activations unq_H
        unq_H = self._get_H(unq_X)

        for n_iter_ in range(self.max_iter):
            # Loop over batches
            for i, (unq_idx, idx) in enumerate(batch_lookup(lookup, n=self.batch_size)):
                if i == n_batch - 1:
                    W_last = self.W_.copy()
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

                if i == n_batch - 1:
                    # Compute the norm of the update of W in the last batch
                    W_change = np.linalg.norm(self.W_ - W_last) / np.linalg.norm(W_last)

            if (W_change < self.tol) and (n_iter_ >= self.min_iter - 1):
                break  # Stop if the change in W is smaller than the tolerance

        # Update self.H_dict_ with the learned encoded vectors (activations)
        self.H_dict_.update(zip(unq_X, unq_H))
        return self

    def get_feature_names(self, n_labels=3, prefix=""):
        """
        Ensures compatibility with sklearn < 1.0.
        Use `get_feature_names_out` instead.
        """
        warnings.warn(
            "Following the changes in scikit-learn 1.0, "
            "get_feature_names is deprecated. "
            "Use get_feature_names_out instead. ",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_feature_names_out(n_labels=n_labels, prefix=prefix)

    def get_feature_names_out(
        self,
        n_labels: int = 3,
        prefix: str = "",
    ) -> List[str]:
        """
        Returns the labels that best summarize the learned components/topics.
        For each topic, labels with the highest activations are selected.

        Parameters
        ----------
        n_labels : int, default=3
            The number of labels used to describe each topic.
        prefix : str, default=""
            Used as a prefix for the categories.

        Returns
        -------
        topic_labels : typing.List[str]
            The labels that best describe each topic.
        """

        vectorizer = CountVectorizer()
        vectorizer.fit(list(self.H_dict_.keys()))
        if parse_version(sklearn_version) < parse_version("1.0"):
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
        topic_labels = [prefix + ", ".join(label) for label in topic_labels]
        return topic_labels

    def score(self, X) -> float:
        """
        Returns the Kullback-Leibler divergence between the n-grams counts
        matrix V of X, and its non-negative factorization HW.

        Parameters
        ----------
        X : array-like (str), shape (n_samples, )
            The data to encode.

        Returns
        -------
        float.
            The Kullback-Leibler divergence.
        """

        # Build n-grams/word counts matrix
        unq_X, lookup = np.unique(X, return_inverse=True)
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

    def partial_fit(self, X, y=None) -> "GapEncoderColumn":
        """
        Partial fit of the GapEncoder on X.
        To be used in an online learning procedure where batches of data are
        coming one by one.

        Parameters
        ----------
        X : array-like, shape (n_samples, )
            The string data to fit the model on.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        GapEncoderColumn
            The fitted GapEncoderColumn instance.
        """

        # Init H_dict_ with empty dict if it's the first call of partial_fit
        if not hasattr(self, "H_dict_"):
            self.H_dict_ = dict()
        # Same thing for the rho_ parameter
        if not hasattr(self, "rho_"):
            self.rho_ = self.rho
        # Check if first item has str or np.str_ type
        assert isinstance(X[0], str), "Input data is not string. "
        # Check if it is not the first batch
        if hasattr(self, "vocabulary"):  # Update unq_X, unq_V with new batch
            unq_X, lookup = np.unique(X, return_inverse=True)
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
            unq_X, unq_V, lookup = self._init_vars(X)

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

    def _add_unseen_keys_to_H_dict(self, X) -> None:
        """
        Add activations of unseen string categories from X to H_dict.
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

    def transform(self, X) -> np.array:
        """
        Return the encoded vectors (activations) H of input strings in X.

        Parameters
        ----------
        X : array-like, shape (n_samples)
            The string data to encode.

        Returns
        -------
        2-d array, shape (n_samples, n_topics)
            Transformed input.
        """
        check_is_fitted(self, "H_dict_")
        # Copy the state of H before continuing fitting it
        pre_trans_H_dict_ = deepcopy(self.H_dict_)
        # Check if the first item has str or np.str_ type
        assert isinstance(X[0], str), "Input data is not string. "
        unq_X = np.unique(X)
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
        feature_names_out = self._get_H(X)
        # Restore H
        self.H_dict_ = pre_trans_H_dict_
        return feature_names_out


class GapEncoder(BaseEstimator, TransformerMixin):
    """Constructs latent topics with continuous encoding.

    This encoder can be understood as a continuous encoding on a set of latent
    categories estimated from the data. The latent categories are built by
    capturing combinations of substrings that frequently co-occur.

    The :class:`~dirty_cat.GapEncoder` supports online learning on batches of
    data for scalability through the :func:`~dirty_cat.GapEncoder.partial_fit`
    method.

    Parameters
    ----------
    n_components : int, optional, default=10
        Number of latent categories used to model string data.
    batch_size : int, optional, default=128
        Number of samples per batch.
    gamma_shape_prior : float, optional, default=1.1
        Shape parameter for the Gamma prior distribution.
    gamma_scale_prior : float, optional, default=1.0
        Scale parameter for the Gamma prior distribution.
    rho : float, optional, default=0.95
        Weight parameter for the update of the *W* matrix.
    rescale_rho : bool, optional, default=False
        If true, use ``rho ** (batch_size / len(X))`` instead of rho to obtain an
        update rate per iteration that is independent of the batch size.
    hashing : bool, optional, default=False
        If true, :class:`~sklearn.feature_extraction.text.HashingVectorizer`
        is used instead of :class:`~sklearn.feature_extraction.text.CountVectorizer`.
        It has the advantage of being very low memory, scalable to large
        datasets as there is no need to store a vocabulary dictionary in
        memory.
    hashing_n_features : int, optional, default=2**12
        Number of features for the :class:`~sklearn.feature_extraction.text.HashingVectorizer`.
        Only relevant if `hashing=True`.
    init : {"k-means++", "random", "k-means"}, optional, default='k-means++'
        Initialization method of the W matrix.
        If `init='k-means++'`, we use the init method of :class:`~sklearn.cluster.KMeans`.
        If `init='random'`, topics are initialized with a Gamma distribution.
        If `init='k-means'`, topics are initialized with a KMeans on the n-grams
        counts. This usually makes convergence faster but is a bit slower.
    tol : float, default=1e-4
        Tolerance for the convergence of the matrix *W*.
    min_iter : int, optional, default=2
        Minimum number of iterations on the input data.
    max_iter : int, optional, default=5
        Maximum number of iterations on the input data.
    ngram_range : int 2-tuple, optional, default=(2, 4)
        The range of ngram length that will be used to build the
        bag-of-n-grams representation of the input data.
    analyzer : {"word", "char", "char_wb"}, optional, default='char'
        Analyzer parameter for the :class:`~sklearn.feature_extraction.text.HashingVectorizer`
        / :class:`~sklearn.feature_extraction.text.CountVectorizer`.
        Describes whether the matrix *V* to factorize should be made of word counts
        or character n-gram counts.
        Option ‘char_wb’ creates character n-grams only from text inside word
        boundaries; n-grams at the edges of words are padded with space.
    add_words : bool, optional, default=False
        If true, add the words counts to the bag-of-n-grams representation
        of the input data.
    random_state : int, :class:`numpy.random.RandomState` or None, optional, default=None
        RNG seed for reproducible output across multiple function calls.
    rescale_W : bool, optional, default=True
        If true, the weight matrix *W* is rescaled at each iteration
        to have a l1 norm equal to 1 for each row.
    max_iter_e_step : int, default=20
        Maximum number of iterations to adjust the activations h at each step.
    handle_missing : {"error", "empty_impute"}, optional, default='empty_impute'
        Whether to raise an error or impute with empty string ``''`` if missing
        values (NaN) are present during fit (default is to impute).
        In the inverse transform, the missing category will be denoted as None.

    Attributes
    ----------
    rho_: float
        Effective update rate for the W matrix
    fitted_models_: list of GapEncoderColumn
        Column-wise fitted GapEncoders
    column_names_: list of str
        Column names of the data the Gap was fitted on

    See Also
    --------
    :class:`~dirty_cat.MinHashEncoder` :
        Encode string columns as a numeric array with the minhash method.
    :class:`~dirty_cat.SimilarityEncoder` :
        Encode string columns as a numeric array with n-gram string similarity.
    :class:`~dirty_cat.deduplicate` :
        Deduplicate data by hierarchically clustering similar strings.

    References
    ----------
    For a detailed description of the method, see
    `Encoding high-cardinality string categorical variables
    <https://hal.inria.fr/hal-02171256v4>`_ by Cerda, Varoquaux (2019).

    Examples
    --------
    >>> enc = GapEncoder(n_components=2)

    Let's encode the following non-normalized data:

    >>> X = [['paris, FR'], ['Paris'], ['London, UK'], ['Paris, France'],
             ['london'], ['London, England'], ['London'], ['Pqris']]

    >>> enc.fit(X)
    GapEncoder(n_components=2)

    The :class:`~dirty_cat.GapEncoder` has found the following two topics:

    >>> enc.get_feature_names_out()
    ['england, london, uk', 'france, paris, pqris']

    It got it right, reccuring topics are "London" and "England" on the
    one side and "Paris" and "France" on the other.

    As this is a continuous encoding, we can look at the level of
    activation of each topic for each category:

    >>> enc.transform(X)
    array([[ 0.05202843, 10.54797156],
          [ 0.05000118,  4.54999882],
          [12.04734788,  0.05265212],
          [ 0.05263068, 16.54736932],
          [ 6.04999624,  0.05000376],
          [19.546716  ,  0.053284  ],
          [ 6.04999623,  0.05000376],
          [ 0.05002016,  4.54997983]])

    The higher the value, the bigger the correspondance with the topic.
    """

    rho_: float
    fitted_models_: List[GapEncoderColumn]
    column_names_: List[str]

    def __init__(
        self,
        n_components: int = 10,
        batch_size: int = 128,
        gamma_shape_prior: float = 1.1,
        gamma_scale_prior: float = 1.0,
        rho: float = 0.95,
        rescale_rho: bool = False,
        hashing: bool = False,
        hashing_n_features: int = 2**12,
        init: Literal["k-means++", "random", "k-means"] = "k-means++",
        tol: float = 1e-4,
        min_iter: int = 2,
        max_iter: int = 5,
        ngram_range: Tuple[int, int] = (2, 4),
        analyzer: Literal["word", "char", "char_wb"] = "char",
        add_words: bool = False,
        random_state: Optional[Union[int, RandomState]] = None,
        rescale_W: bool = True,
        max_iter_e_step: int = 20,
        handle_missing: Literal["error", "empty_impute"] = "zero_impute",
    ):
        self.ngram_range = ngram_range
        self.n_components = n_components
        self.gamma_shape_prior = gamma_shape_prior  # 'a' parameter
        self.gamma_scale_prior = gamma_scale_prior  # 'b' parameter
        self.rho = rho
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

    def _more_tags(self) -> Dict[str, List[str]]:
        """
        Used internally by sklearn to ease the estimator checks.
        """
        return {"X_types": ["categorical"]}

    def _create_column_gap_encoder(self) -> GapEncoderColumn:
        return GapEncoderColumn(
            ngram_range=self.ngram_range,
            n_components=self.n_components,
            analyzer=self.analyzer,
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
        if self.handle_missing not in ["error", "zero_impute"]:
            raise ValueError(
                "handle_missing should be either 'error' or "
                f"'zero_impute', got {self.handle_missing!r}. "
            )

        missing_mask = _object_dtype_isnan(X)

        if missing_mask.any():
            if self.handle_missing == "error":
                raise ValueError("Input data contains missing values. ")
            elif self.handle_missing == "zero_impute":
                X[missing_mask] = ""

        return X

    def fit(self, X, y=None) -> "GapEncoder":
        """
        Fit the instance on batches of X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The string data to fit the model on.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        :class:`~dirty_cat.GapEncoder`
            Fitted :class:`~dirty_cat.GapEncoder` instance (self).
        """

        # Check that n_samples >= n_components
        if len(X) < self.n_components:
            raise ValueError(
                f"n_samples={len(X)} should be >= n_components={self.n_components}. "
            )
        # Copy parameter rho
        self.rho_ = self.rho
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

    def transform(self, X) -> np.array:
        """
        Return the encoded vectors (activations) H of input strings in X.
        Given the learnt topics W, the activations H are tuned to fit V = HW.
        When X has several columns, they are encoded separately and
        then concatenated.

        Remark: calling transform multiple times in a row on the same
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
        check_is_fitted(self, "fitted_models_")
        # Check input data shape
        X = check_input(X)
        X = self._handle_missing(X)
        X_enc = []
        for k in range(X.shape[1]):
            X_enc.append(self.fitted_models_[k].transform(X[:, k]))
        X_enc = np.hstack(X_enc)
        return X_enc

    def partial_fit(self, X, y=None) -> "GapEncoder":
        """
        Partial fit of the GapEncoder on X.
        To be used in an online learning procedure where batches of data are
        coming one by one.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The string data to fit the model on.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        GapEncoder
            Fitted GapEncoder instance.
        """

        # If X is a dataframe, store its column names
        if isinstance(X, pd.DataFrame):
            self.column_names_ = list(X.columns)
        # Check input data shape
        X = check_input(X)
        X = self._handle_missing(X)
        # Init the `GapEncoderColumn` instances if the model was
        # not fitted already.
        if not hasattr(self, "fitted_models_"):
            self.fitted_models_ = [
                self._create_column_gap_encoder() for _ in range(X.shape[1])
            ]
        for k in range(X.shape[1]):
            self.fitted_models_[k].partial_fit(X[:, k])
        return self

    def get_feature_names_out(
        self,
        col_names: Optional[Union[Literal["auto"], List[str]]] = None,
        n_labels: int = 3,
    ):
        """
        Returns the labels that best summarize the learned components/topics.
        For each topic, labels with the highest activations are selected.

        Parameters
        ----------
        col_names : "auto" or list of strings, optional
            The column names to be added as prefixes before the labels.
            If col_names == None, no prefixes are used.
            If col_names == 'auto', column names are automatically defined:

                - if the input data was a dataframe, its column names are used,
                - otherwise, 'col1', ..., 'colN' are used as prefixes.

            Prefixes can be manually set by passing a list for col_names.
        n_labels : int, default=3
            The number of labels used to describe each topic.

        Returns
        -------
        list of strings
            The labels that best describe each topic.
        """
        check_is_fitted(self, "fitted_models_")
        # Generate prefixes
        if isinstance(col_names, str) and col_names == "auto":
            if hasattr(self, "column_names_"):  # Use column names
                prefixes = ["%s: " % col for col in self.column_names_]
            else:  # Use 'col1: ', ... 'colN: ' as prefixes
                prefixes = ["col%d: " % i for i in range(len(self.fitted_models_))]
        elif col_names is None:  # Empty prefixes
            prefixes = [""] * len(self.fitted_models_)
        else:
            prefixes = ["%s: " % col for col in col_names]
        labels = list()
        for k, enc in enumerate(self.fitted_models_):
            col_labels = enc.get_feature_names_out(n_labels, prefixes[k])
            labels.extend(col_labels)
        return labels

    def get_feature_names(
        self, input_features=None, col_names: List[str] = None, n_labels: int = 3
    ) -> List[str]:
        """
        Ensures compatibility with sklearn < 1.0.
        Use `get_feature_names_out` instead.
        """
        if parse_version(sklearn_version) >= parse_version("1.0"):
            warnings.warn(
                "Following the changes in scikit-learn 1.0, "
                "get_feature_names is deprecated. "
                "Use get_feature_names_out instead. ",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.get_feature_names_out(col_names, n_labels)

    def score(self, X) -> float:
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
        float.
            The Kullback-Leibler divergence.
        """
        X = check_input(X)
        kl_divergence = 0
        for k in range(X.shape[1]):
            kl_divergence += self.fitted_models_[k].score(X[:, k])
        return kl_divergence


def _rescale_W(W: np.array, A: np.array) -> None:
    """
    Rescale the topics W to have a L1-norm equal to 1.
    Note that they are modified in-place.
    """
    s = W.sum(axis=1, keepdims=True)
    W /= s
    A /= s


def _multiplicative_update_w(
    Vt: np.array,
    W: np.array,
    A: np.array,
    B: np.array,
    Ht: np.array,
    rescale_W: bool,
    rho: float,
) -> Tuple[np.array, np.array, np.array]:
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


def _rescale_h(V: np.array, H: np.array) -> np.array:
    """
    Rescale the activations H.
    """
    epsilon = 1e-10  # in case of a document having length=0
    H *= np.maximum(epsilon, V.sum(axis=1).A)
    H /= H.sum(axis=1, keepdims=True)
    return H


def _multiplicative_update_h(
    Vt: np.array,
    W: np.array,
    Ht: np.array,
    epsilon: float = 1e-3,
    max_iter: int = 10,
    rescale_W: bool = False,
    gamma_shape_prior: float = 1.1,
    gamma_scale_prior: float = 1.0,
):
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
            squared_norm = np.dot(ht_out - ht, ht_out - ht) / np.dot(ht, ht)
            ht[:] = ht_out
    return Ht


def batch_lookup(
    lookup: np.array,
    n: int = 1,
) -> Generator[Tuple[np.array, np.array], None, None]:
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
    n_prototypes: int,
    analyzer: Literal["word", "char", "char_wb"] = "char",
    hashing_dim: int = 128,
    ngram_range: Tuple[int, int] = (2, 4),
    sparse: bool = False,
    sample_weight=None,
    random_state: Optional[Union[int, RandomState]] = None,
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
    kmeans = KMeans(n_clusters=n_prototypes, random_state=random_state)
    kmeans.fit(projected, sample_weight=sample_weight)
    centers = kmeans.cluster_centers_
    neighbors = NearestNeighbors()
    neighbors.fit(projected)
    indexes_prototypes = np.unique(neighbors.kneighbors(centers, 1)[-1])
    return np.sort(X[indexes_prototypes])
