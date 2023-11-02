"""
Implements the SimilarityEncoder, a generalization of the OneHotEncoder,
which encodes similarity instead of equality of values.
"""

from typing import Literal

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import parse_version
from sklearn.utils.validation import check_is_fitted

from ._string_distances import get_ngram_count, preprocess

# Ignore lines too long, first docstring lines can't be cut
# flake8: noqa: E501


def _ngram_similarity_one_sample_inplace(
    x_count_vector: NDArray,
    vocabulary_count_matrix: NDArray,
    str_x: str,
    vocabulary_ngram_counts: NDArray,
    se_dict: dict,
    unq_X: NDArray,
    i: int,
    ngram_range: tuple[int, int],
) -> None:
    """
    Update inplace a dict of similarities between a string and a vocabulary

    Parameters
    ----------
    x_count_vector : ndarray
        Count vector of the sample based on the ngrams of the vocabulary
    vocabulary_count_matrix : ndarray
        Count vector of the vocabulary based on its ngrams
    str_x: str
        The actual sample string
    vocabulary_ngram_counts : ndarray
        Number of ngrams for each unique element of the vocabulary
    se_dict : dict
        Dictionary containing the similarities for each x in unq_X
    unq_X : ndarray
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
    cats: list[str],
    ngram_range: tuple[int, int],
    analyzer: Literal["word", "char", "char_wb"],
    hashing_dim: int,
    dtype: type = np.float64,
) -> NDArray:
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
            analyzer=analyzer, ngram_range=(min_n, max_n), dtype=dtype
        )
        vectorizer.fit(np.concatenate((cats, unq_X_)))
    else:
        vectorizer = HashingVectorizer(
            analyzer=analyzer,
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


class SimilarityEncoder(OneHotEncoder):
    """Encode string categories to a similarity matrix, to capture fuzziness across a few categories.

    The input to this transformer should be an array-like of strings.
    The method is based on calculating the morphological similarities
    between the categories.
    This encoding is an alternative to OneHotEncoder for
    dirty categorical variables.

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
       These prototypes can be provided by the user. Otherwise, we recommend
       using the MinHashEncoder or GapEncoder when taking all unique entries
       leads to too many prototypes.

    The similarity measure is based on the proportion of common n-grams between
    two strings.

    Parameters
    ----------
    ngram_range : int 2-tuple (min_n, max_n), default=(2, 4)
        The lower and upper boundaries of the range of n-values for different
        n-grams used in the string similarity. All values of `n` such
        that ``min_n <= n <= max_n`` will be used.
    analyzer : {'word', 'char', 'char_wb'}, default='char'
        Analyzer parameter for the HashingVectorizer / CountVectorizer.
        Describes whether the matrix `V` to factorize should be made of
        word counts or character-level n-gram counts.
        Option ‘char_wb’ creates character n-grams only from text inside word
        boundaries; n-grams at the edges of words are padded with space.
    categories : {'auto'} or list of list of str
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : `categories[i]` holds the categories expected in the i-th
          column. The passed categories must be sorted and should not mix
          strings and numeric values.

        The categories used can be found in the SimilarityEncoder.categories_
        attribute.
    dtype : number type, default=float64
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
        "Missing values" are any value for which ``pandas.isna`` returns
        ``True``, such as ``numpy.nan`` or ``None``.
    hashing_dim : int, optional
        If `None`, the base vectorizer is a CountVectorizer, otherwise it is a
        HashingVectorizer with a number of features equal to `hashing_dim`.
    n_jobs : int, optional
        Maximum number of processes used to compute similarity matrices. Used
        only if `fast=True` in SimilarityEncoder.transform.

    Attributes
    ----------
    categories_ : list of ndarray
        The categories of each feature determined during fitting
        (in the same order as the output of SimilarityEncoder.transform).

    See Also
    --------
    MinHashEncoder :
        Encode string columns as a numeric array with the minhash method.
    GapEncoder :
        Encodes dirty categories (strings) by constructing latent topics
        with continuous encoding.
    deduplicate :
        Deduplicate data by hierarchically clustering similar strings.

    Notes
    -----
    The functionality of SimilarityEncoder is easy to explain and understand,
    but it is not scalable. It is useful only to capture links across a few categories
    (eg eg: "west", "north", "north-west"), but not when there are many categories,
    as with open-ended entries.
    Instead, the GapEncoder is usually recommended.

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

    It inherits the same methods as theOneHotEncoder:

    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]

    But it provides a continuous encoding based on similarity
    instead of a discrete one based on exact matches:

    >>> enc.transform([['Female', 1], ['Male', 4]])
    array([[1.        , 0.42..., 1.        , 0.        , 0.        ],
           [0.42..., 1.        , 0.        , 0.        , 0.        ]])

    >>> enc.get_feature_names_out(['gender', 'group'])
    array(['gender_Female', 'gender_Male', 'group_1', 'group_2', 'group_3'],
          dtype=object)
    """

    categories_: list[NDArray]
    n_features_in_: int
    drop_idx_: NDArray
    vectorizers_: list[CountVectorizer]
    vocabulary_count_matrices_: list[NDArray]
    vocabulary_ngram_counts_: list[list[int]]
    _infrequent_enabled: bool

    def __init__(
        self,
        *,
        ngram_range: tuple[int, int] = (2, 4),
        analyzer: Literal["word", "char", "char_wb"] = "char",
        categories: Literal["auto"] | list[list[str]] = "auto",
        dtype: type = np.float64,
        handle_unknown: Literal["error", "ignore"] = "ignore",
        handle_missing: Literal["error", ""] = "",
        hashing_dim: int | None = None,
        n_jobs: int | None = None,
    ):
        super().__init__()
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.hashing_dim = hashing_dim
        self.n_jobs = n_jobs

        if not isinstance(categories, list):
            if categories not in ["auto"]:
                raise ValueError(
                    f"Got categories={self.categories}, but expected "
                    "'auto' or a list of prototypes. "
                )

    def fit(self, X: ArrayLike, y=None) -> "SimilarityEncoder":
        """Fit the instance to `X`.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        SimilarityEncoder
            The fitted SimilarityEncoder instance (self).
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
            mask = pd.isna(X)
            if X.dtype.kind == "O" and mask.any():
                if self.handle_missing == "error":
                    raise ValueError(
                        "Found missing values in input data; set "
                        "handle_missing='' to encode with missing values. "
                    )
                else:
                    X[mask] = self.handle_missing

        Xlist, n_samples, n_features = self._check_X(X)
        self._check_n_features(X, reset=True)

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

        if self.categories not in ["auto"]:
            for cats in self.categories:
                if not np.all(np.sort(cats) == np.array(cats)):
                    raise ValueError("Unsorted categories are not yet supported. ")

        self.categories_ = list()

        for i in range(n_features):
            Xi = Xlist[i]
            if self.categories == "auto":
                self.categories_.append(np.unique(Xi))
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
                ngram_range=self.ngram_range,
                analyzer=self.analyzer,
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

        self._infrequent_enabled = False
        if parse_version(sklearn.__version__) >= parse_version("1.2.2"):
            self.drop_idx_ = self._set_drop_idx()
        else:
            self.drop_idx_ = self._compute_drop_idx()

        self._n_features_outs = list(map(len, self.categories_))
        return self

    def transform(self, X: ArrayLike, fast: bool = True) -> NDArray:
        """Transform `X` using specified encoding scheme.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        fast : bool, default=True
            Whether to use the fast computation of ngrams.

        Returns
        -------
        ndarray, shape [n_samples, n_features_new]
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
            mask = pd.isna(X)
            if X.dtype.kind == "O" and mask.any():
                if self.handle_missing == "error":
                    raise ValueError(
                        "Found missing values in input data; set "
                        "handle_missing='' to encode with missing values. "
                    )
                else:
                    X[mask] = self.handle_missing

        Xlist, n_samples, n_features = self._check_X(X)
        self._check_n_features(X, reset=False)

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
        out = np.empty((n_samples, total_length), dtype=self.dtype)
        last = 0
        for j, categories in enumerate(self.categories_):
            if fast:
                encoded_Xj = self._ngram_similarity_fast(Xlist[j], j)
            else:
                encoded_Xj = ngram_similarity_matrix(
                    Xlist[j],
                    categories,
                    ngram_range=(min_n, max_n),
                    analyzer=self.analyzer,
                    hashing_dim=self.hashing_dim,
                    dtype=np.float32,
                )

            out[:, last : last + len(categories)] = encoded_Xj
            last += len(categories)
        return out

    def _ngram_similarity_fast(
        self,
        X: list | NDArray,
        col_idx: int,
    ) -> NDArray:
        """
        Fast computation of ngram similarity.

        SimilarityEncoder.transform uses the count vectors
        of the vocabulary in its computations.
        In `ngram_similarity`, these count vectors have to be
        re-computed each time, which can slow down the execution. In this
        method, the count vectors are recovered from
        :attr:`~skrub.SimilarityEncoder.vocabulary_count_matrices`,
        speeding up the execution.

        Parameters
        ----------
        X : list or ndarray
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

    def _more_tags(self):
        return {
            "X_types": ["2darray", "categorical", "string"],
            "preserves_dtype": [],
            "allow_nan": True,
            "_xfail_checks": {
                "check_estimator_sparse_data": (
                    "Cannot create sparse matrix with strings."
                ),
                "check_estimators_dtypes": "We only support string dtypes.",
            },
        }
