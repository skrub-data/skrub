"""
Implements the SimilarityEncoder, a generalization of the OneHotEncoder,
which encodes similarity instead of equality of values.
"""

import warnings
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import sklearn
from joblib import Parallel, delayed
from numpy.random import RandomState
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state
from sklearn.utils.fixes import _object_dtype_isnan
from sklearn.utils.validation import check_is_fitted

from ._string_distances import get_ngram_count, preprocess
from ._utils import parse_version


def _ngram_similarity_one_sample_inplace(
    x_count_vector: np.ndarray,
    vocabulary_count_matrix: np.ndarray,
    str_x: str,
    vocabulary_ngram_counts: np.ndarray,
    se_dict: dict,
    unq_X: np.ndarray,
    i: int,
    ngram_range: Tuple[int, int],
) -> None:
    """
    Update inplace a dict of similarities between a string and a vocabulary

    Parameters
    ----------
    x_count_vector : :obj:`~numpy.ndarray`
        Count vector of the sample based on the ngrams of the vocabulary
    vocabulary_count_matrix : :obj:`~numpy.ndarray`
        Count vector of the vocabulary based on its ngrams
    str_x: str
        The actual sample string
    vocabulary_ngram_counts : :obj:`~numpy.ndarray`
        Number of ngrams for each unique element of the vocabulary
    se_dict : dict
        Dictionary containing the similarities for each x in unq_X
    unq_X : :obj:`~numpy.ndarray`
        The arrays of all unique samples
    i : str
        The index of x_count_vector in the csr count matrix
    ngram_range : 2-tuple of int
        The lower and upper boundaries of the range of n-values for different
        n-grams used in the string similarity. All values of `n` such
        that ``min_n <= n <= max_n`` will be used.
    """
    nonzero_idx = x_count_vector.indices
    nonzero_vals = x_count_vector.data

    same_grams = np.asarray(
        (vocabulary_count_matrix[:, nonzero_idx].minimum(nonzero_vals)).sum(axis=1)
    )

    all_grams = (
        get_ngram_count(str_x, ngram_range) + vocabulary_ngram_counts - same_grams
    )
    similarity = np.divide(
        same_grams, all_grams, out=np.zeros_like(same_grams), where=all_grams != 0
    )
    se_dict[unq_X[i]] = similarity.reshape(-1)


def ngram_similarity_matrix(
    X,
    cats: List[str],
    ngram_range: Tuple[int, int],
    hashing_dim: int,
    dtype: type = np.float64,
) -> np.ndarray:
    """
    Similarity encoding for dirty categorical variables:
    Given two arrays of strings, returns the similarity encoding matrix
    of size len(X) x len(cats)

    ngram_sim(s_i, s_j) =
        ||min(ci, cj)||_1 / (||ci||_1 + ||cj||_1 - ||min(ci, cj)||_1)
    """
    min_n, max_n = ngram_range
    unq_X = np.unique(X)
    cats = np.array([" %s " % cat for cat in cats])
    unq_X_ = np.array([" %s " % x for x in unq_X])
    if not hashing_dim:
        vectorizer = CountVectorizer(
            analyzer="char", ngram_range=(min_n, max_n), dtype=dtype
        )
        vectorizer.fit(np.concatenate((cats, unq_X_)))
    else:
        vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(min_n, max_n),
            n_features=hashing_dim,
            norm=None,
            alternate_sign=False,
            dtype=dtype,
        )
        vectorizer.fit(X)
    count_cats = vectorizer.transform(cats)
    count_X = vectorizer.transform(unq_X_)
    # We don't need the vectorizer anymore, delete it to save memory
    del vectorizer
    sum_cats = np.asarray(count_cats.sum(axis=1))
    SE_dict = {}

    for i, x in enumerate(count_X):
        _, nonzero_idx, nonzero_vals = sparse.find(x)
        same_grams = np.asarray(
            (count_cats[:, nonzero_idx].minimum(nonzero_vals)).sum(axis=1)
        )
        all_grams = x.sum() + sum_cats - same_grams
        similarity = np.divide(same_grams, all_grams)
        SE_dict[unq_X[i]] = similarity.reshape(-1)
    # We don't need the counts anymore, delete them to save memory
    del count_cats, count_X

    out = np.empty((len(X), similarity.size), dtype=dtype)
    for x, out_row in zip(X, out):
        out_row[:] = SE_dict[x]

    return np.nan_to_num(out, copy=False)


def get_prototype_frequencies(prototypes: np.ndarray) -> np.ndarray:
    """
    Computes the frequencies of the values contained in prototypes
    Reverse sorts the array by the frequency
    Returns a numpy array of the values without their frequencies
    """
    uniques, counts = np.unique(prototypes, return_counts=True)
    sorted_indexes = np.argsort(counts)[::-1]
    return uniques[sorted_indexes], counts[sorted_indexes]


def get_kmeans_prototypes(
    X,
    n_prototypes: int,
    hashing_dim: int = 128,
    ngram_range: Tuple[int, int] = (3, 3),
    sparse: bool = False,
    sample_weight=None,
    random_state: Optional[Union[int, RandomState]] = None,
) -> np.ndarray:
    """
    Computes prototypes based on:
      - dimensionality reduction (via hashing n-grams)
      - k-means clustering
      - nearest neighbor
    """
    vectorizer = HashingVectorizer(
        analyzer="char",
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
    if indexes_prototypes.shape[0] < n_prototypes:
        warnings.warn(
            "Final number of unique prototypes is lower than "
            + "n_prototypes (expected). "
        )
    return np.sort(X[indexes_prototypes])


class SimilarityEncoder(OneHotEncoder):
    """Encode string categorical features to a similarity matrix.

    The input to this transformer should be an array-like of strings.
    The method is based on calculating the morphological similarities
    between the categories.
    This encoding is an alternative to
    :class:`~sklearn.preprocessing.OneHotEncoder` for dirty categorical variables.

    The principle of this encoder is as follows:

    1. Given an input string array ``X = [x1, ..., xn]`` with `k` unique
       categories ``[c1, ..., ck]`` and a similarity measure ``sim(s1, s2)``
       between strings, we define the encoded vector of `xi` as
       ``[sim(xi, c1), ... , sim(xi, ck)]``.
       Similarity encoding of `X` results in a matrix with shape (`n`, `k`)
       that captures morphological similarities between string entries.
    2. To avoid dealing with high-dimensional encodings when `k` is high,
       we can use ``d << k`` prototypes ``[p1, ..., pd]`` with which
       similarities will be computed:  ``xi -> [sim(xi, p1), ..., sim(xi, pd)]``.
       These prototypes can be automatically sampled from the input data
       (most frequent categories, KMeans) or provided by the user.

    Parameters
    ----------
    similarity : None
        Deprecated in skrub 0.3, will be removed in 0.5.
        Was used to specify the type of pairwise string similarity to use.
        Since 0.3, only the ngram similarity is supported.
    ngram_range : int 2-tuple (min_n, max_n), default=(2, 4)
        The lower and upper boundaries of the range of n-values for different
        n-grams used in the string similarity. All values of `n` such
        that ``min_n <= n <= max_n`` will be used.
    categories : {'auto', 'k-means', 'most_frequent'} or list of list of str
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : `categories[i]` holds the categories expected in the i-th
          column. The passed categories must be sorted and should not mix
          strings and numeric values.
        - 'most_frequent' : Computes the most frequent values for every
           categorical variable
        - 'k-means' : Computes the K nearest neighbors of K-mean centroids
           in order to choose the prototype categories

        The categories used can be found in the
        :attr:`~SimilarityEncoder.categories_` attribute.
    dtype : number type, default :class:`~numpy.float64`
        Desired dtype of output.
    handle_unknown : 'error' or 'ignore', default=''
        Whether to raise an error or ignore if an unknown categorical feature
        is present during transform (default is to ignore). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting encoded columns for this feature
        will be all zeros. In the inverse transform, an unknown category
        will be denoted as None.
    handle_missing : 'error' or '', default=''
        Whether to raise an error or impute with blank string '' if missing
        values (NaN) are present during fit (default is to impute).
        When this parameter is set to '', and a missing value is encountered
        during fit_transform, the resulting encoded columns for this feature
        will be all zeros. In the inverse transform, the missing category
        will be denoted as None.
    hashing_dim : int, optional
        If `None`, the base vectorizer is a
        :obj:`~sklearn.feature_extraction.text.CountVectorizer`,
        otherwise it is a
        :obj:`~sklearn.feature_extraction.text.HashingVectorizer`
        with a number of features equal to `hashing_dim`.
    n_prototypes : int, optional
        Useful when `most_frequent` or `k-means` is used.
        Must be a positive integer.
    random_state : int or RandomState, optional
        Useful when `k-means` strategy is used.
    n_jobs : int, optional
        Maximum number of processes used to compute similarity matrices. Used
        only if `fast=True` in :func:`~SimilarityEncoder.transform`.

    Attributes
    ----------
    categories_ : list of :obj:`~numpy.ndarray`
        The categories of each feature determined during fitting
        (in the same order as the output of :func:`~SimilarityEncoder.transform`).

    See Also
    --------
    :class:`skrub.MinHashEncoder` :
        Encode string columns as a numeric array with the minhash method.
    :class:`skrub.GapEncoder` :
        Encodes dirty categories (strings) by constructing latent topics
        with continuous encoding.
    :class:`skrub.deduplicate` :
        Deduplicate data by hierarchically clustering similar strings.

    Notes
    -----
    The functionality of :class:`SimilarityEncoder` is easy to explain
    and understand, but it is not scalable.
    Instead, the :class:`~skrub.GapEncoder` is usually recommended.

    References
    ----------
    For a detailed description of the method, see
    `Similarity encoding for learning with dirty categorical variables
    <https://hal.inria.fr/hal-01806175>`_ by Cerda, Varoquaux, Kegl. 2018
    (Machine Learning journal, Springer).

    Examples
    --------
    >>> enc = SimilarityEncoder()
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    SimilarityEncoder()

    It inherits the same methods as the
    :class:`~sklearn.preprocessing.OneHotEncoder`:

    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]

    But it provides a continuous encoding based on similarity
    instead of a discrete one based on exact matches:

    >>> enc.transform([['Female', 1], ['Male', 4]])
    array([[1., 0.42857143, 1., 0., 0.],
           [0.42857143, 1., 0. , 0. , 0.]])

    >>> enc.inverse_transform(
    >>>     [[1., 0.42857143, 1., 0., 0.], [0.42857143, 1., 0. , 0. , 0.]]
    >>> )
    array([['Female', 1],
           ['Male', None]], dtype=object)

    >>> enc.get_feature_names_out(['gender', 'group'])
    array(['gender_Female', 'gender_Male', 'group_1', 'group_2', 'group_3'], ...)
    """

    categories_: List[np.ndarray]
    n_features_in_: int
    random_state_: Union[int, RandomState]
    drop_idx_: np.ndarray
    vectorizers_: List[CountVectorizer]
    vocabulary_count_matrices_: List[np.ndarray]
    vocabulary_ngram_counts_: List[List[int]]
    _infrequent_enabled: bool

    def __init__(
        self,
        similarity: str = None,
        ngram_range: Tuple[int, int] = (2, 4),
        categories: Union[
            Literal["auto", "k-means", "most_frequent"], List[List[str]]
        ] = "auto",
        dtype: type = np.float64,
        handle_unknown: Literal["error", "ignore"] = "ignore",
        handle_missing: Literal["error", ""] = "",
        hashing_dim: Optional[int] = None,
        n_prototypes: Optional[int] = None,
        random_state: Optional[Union[int, RandomState]] = None,
        n_jobs: Optional[int] = None,
    ):
        super().__init__()
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.ngram_range = ngram_range
        self.hashing_dim = hashing_dim
        self.n_prototypes = n_prototypes
        self.random_state = random_state
        self.n_jobs = n_jobs

        if similarity is not None:
            warnings.warn(
                'The "similarity" argument is deprecated since skrub 0.3, '
                "and will be removed in 0.5."
                "The n-gram similarity is the only one currently supported. ",
                category=UserWarning,
                stacklevel=2,
            )
        self.similarity = None

        if not isinstance(categories, list):
            if categories not in ["auto", "k-means", "most_frequent"]:
                raise ValueError(
                    f"Got categories={self.categories}, but expected "
                    "any of {'auto', 'k-means', 'most_frequent'}. "
                )
        if categories in ["k-means", "most_frequent"] and (
            n_prototypes is None or n_prototypes == 0
        ):
            raise ValueError(
                "n_prototypes expected None or a positive non null integer. "
            )
        if categories == "auto" and n_prototypes is not None:
            warnings.warn('n_prototypes parameter ignored with category type "auto". ')

    def get_most_frequent(self, prototypes: List[str]) -> np.ndarray:
        """Get the most frequent category prototypes.

        Parameters
        ----------
        prototypes : list of str
            The list of values for a category variable.

        Returns
        -------
        :obj:`~numpy.ndarray`
            The n_prototypes most frequent values for a category variable.
        """
        values, _ = get_prototype_frequencies(prototypes)
        return values[: self.n_prototypes]

    def fit(self, X, y=None) -> "SimilarityEncoder":
        """Fit the instance to `X`.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        :obj:`SimilarityEncoder`
            The fitted :class:`SimilarityEncoder` instance (self).
        """

        if self.handle_missing not in ["error", ""]:
            raise ValueError(
                f"Got handle_missing={self.handle_missing}, but expected "
                "any of {'error', ''}. "
            )
        if hasattr(X, "iloc") and X.isna().values.any():
            if self.handle_missing == "error":
                raise ValueError(
                    "Found missing values in input data; set "
                    "handle_missing='' to encode with missing values. "
                )
            else:
                X = X.fillna(self.handle_missing)
        elif not hasattr(X, "dtype") and isinstance(X, list):
            X = np.asarray(X, dtype=object)

        if hasattr(X, "dtype"):
            mask = _object_dtype_isnan(X)
            if X.dtype.kind == "O" and mask.any():
                if self.handle_missing == "error":
                    raise ValueError(
                        "Found missing values in input data; set "
                        "handle_missing='' to encode with missing values. "
                    )
                else:
                    X[mask] = self.handle_missing

        Xlist, n_samples, n_features = self._check_X(X)
        self.n_features_in_ = n_features

        if self.handle_unknown not in ["error", "ignore"]:
            raise ValueError(
                f"Got handle_unknown={self.handle_unknown!r}, but expected "
                "any of {'error', 'ignore'}. "
            )

        if (self.hashing_dim is not None) and (not isinstance(self.hashing_dim, int)):
            raise ValueError(
                f"Got hashing_dim={self.hashing_dim!r}, which has an invalid "
                f"type ({type(self.hashing_dim)}), expected None or int. "
            )

        if self.categories not in ["auto", "most_frequent", "k-means"]:
            for cats in self.categories:
                if not np.all(np.sort(cats) == np.array(cats)):
                    raise ValueError("Unsorted categories are not yet supported. ")

        self.categories_ = list()
        self.random_state_ = check_random_state(self.random_state)

        for i in range(n_features):
            Xi = Xlist[i]
            if self.categories == "auto":
                self.categories_.append(np.unique(Xi))
            elif self.categories == "most_frequent":
                self.categories_.append(self.get_most_frequent(Xi))
            elif self.categories == "k-means":
                uniques, count = np.unique(Xi, return_counts=True)
                self.categories_.append(
                    get_kmeans_prototypes(
                        uniques,
                        self.n_prototypes,
                        sample_weight=count,
                        random_state=self.random_state_,
                    )
                )
            else:
                if self.handle_unknown == "error":
                    valid_mask = np.in1d(Xi, self.categories[i])
                    if not np.all(valid_mask):
                        diff = np.unique(Xi[~valid_mask])
                        raise ValueError(
                            f"Found unknown categories {diff} in column {i} "
                            "during fit. "
                        )
                self.categories_.append(np.array(self.categories[i], dtype=object))

        self.vectorizers_ = []
        self.vocabulary_count_matrices_ = []
        self.vocabulary_ngram_counts_ = []

        for i in range(n_features):
            vectorizer = CountVectorizer(
                analyzer="char",
                ngram_range=self.ngram_range,
                dtype=self.dtype,
                strip_accents=None,
            )

            # Store the raw-categories (and not the preprocessed
            # categories) but use the preprocessed categories to compute
            # the stored count_matrices. This done to preserve the
            # equivalency between the user input and the categories_
            # attribute of the SimilarityEncoder, while being compliant
            # with the CountVectorizer preprocessing steps.
            categories = self.categories_[i]

            self.vectorizers_.append(vectorizer)

            self.vocabulary_count_matrices_.append(
                vectorizer.fit_transform(
                    [preprocess(category) for category in categories]
                )
            )

            self.vocabulary_ngram_counts_.append(
                [
                    get_ngram_count(preprocess(category), self.ngram_range)
                    for category in categories
                ]
            )

        if parse_version(sklearn.__version__) >= parse_version("1.1.0"):
            self._infrequent_enabled = False
        if parse_version(sklearn.__version__) >= parse_version("1.2.2"):
            self.drop_idx_ = self._set_drop_idx()
        else:
            self.drop_idx_ = self._compute_drop_idx()

        return self

    def transform(self, X, fast: bool = True) -> np.ndarray:
        """Transform `X` using specified encoding scheme.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        fast : bool, default=True
            Whether to use the fast computation of ngrams.

        Returns
        -------
        :obj:`~numpy.ndarray`, shape [n_samples, n_features_new]
            Transformed input.
        """
        check_is_fitted(self, "categories_")
        if hasattr(X, "iloc") and X.isna().values.any():
            if self.handle_missing == "error":
                raise ValueError(
                    "Found missing values in input data; set "
                    "handle_missing='' to encode with missing values. "
                )
            else:
                X = X.fillna(self.handle_missing)
        elif not hasattr(X, "dtype") and isinstance(X, list):
            X = np.asarray(X, dtype=object)

        if hasattr(X, "dtype"):
            mask = _object_dtype_isnan(X)
            if X.dtype.kind == "O" and mask.any():
                if self.handle_missing == "error":
                    raise ValueError(
                        "Found missing values in input data; set "
                        "handle_missing='' to encode with missing values. "
                    )
                else:
                    X[mask] = self.handle_missing

        Xlist, n_samples, n_features = self._check_X(X)

        for i in range(n_features):
            Xi = Xlist[i]
            valid_mask = np.in1d(Xi, self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == "error":
                    diff = np.unique(X[~valid_mask, i])
                    raise ValueError(
                        f"Found unknown categories {diff} in column {i} during fit. "
                    )

        min_n, max_n = self.ngram_range

        total_length = sum(len(x) for x in self.categories_)
        out = np.empty((len(X), total_length), dtype=self.dtype)
        last = 0
        for j, categories in enumerate(self.categories_):
            if fast:
                encoded_Xj = self._ngram_similarity_fast(Xlist[j], j)
            else:
                encoded_Xj = ngram_similarity_matrix(
                    Xlist[j],
                    categories,
                    ngram_range=(min_n, max_n),
                    hashing_dim=self.hashing_dim,
                    dtype=np.float32,
                )

            out[:, last : last + len(categories)] = encoded_Xj
            last += len(categories)
        return out

    def _ngram_similarity_fast(
        self,
        X: Union[list, np.ndarray],
        col_idx: int,
    ) -> np.ndarray:
        """
        Fast computation of ngram similarity.

        :func:`~skrub.SimilarityEncoder.transform` uses the count vectors
        of the vocabulary in its computations.
        In `ngram_similarity`, these count vectors have to be
        re-computed each time, which can slow down the execution. In this
        method, the count vectors are recovered from
        :attr:`~skrub.SimilarityEncoder.vocabulary_count_matrices`,
        speeding up the execution.

        Parameters
        ----------
        X : list or :obj:`numpy.ndarray`
            Observations being transformed.
        col_idx : int
            The column index of X in the original feature matrix.
        """
        vectorizer = self.vectorizers_[col_idx]

        unq_X = np.unique(X)
        unq_X_ = np.array([preprocess(x) for x in unq_X])

        X_count_matrix = vectorizer.transform(unq_X_)
        vocabulary_count_matrix = self.vocabulary_count_matrices_[col_idx]
        vocabulary_ngram_count = np.array(
            self.vocabulary_ngram_counts_[col_idx]
        ).reshape(-1, 1)

        se_dict = {}

        Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(_ngram_similarity_one_sample_inplace)(
                X_count_vector,
                vocabulary_count_matrix,
                x_str,
                vocabulary_ngram_count,
                se_dict,
                unq_X,
                i,
                self.ngram_range,
            )
            for X_count_vector, x_str, i in zip(
                X_count_matrix, unq_X_, range(len(unq_X))
            )
        )

        out = np.empty(
            (len(X), vocabulary_count_matrix.shape[0]),
            dtype=self.dtype,
        )

        for x, out_row in zip(X, out):
            out_row[:] = se_dict[x]

        return np.nan_to_num(out, copy=False)
