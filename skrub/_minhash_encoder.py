"""
Implements the MinHashEncoder, which encodes string categorical features by
applying the MinHash method to n-gram decompositions of strings.
"""
from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Literal

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import gen_even_slices, murmurhash3_32
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted

from ._fast_hash import ngram_min_hash
from ._string_distances import get_unique_ngrams
from ._utils import LRUDict, check_input

NoneType = type(None)


class MinHashEncoder(TransformerMixin, BaseEstimator):
    """Encode string categorical features by applying the MinHash method to n-gram \
    decompositions of strings.

    The principle is as follows:

    1. A string is viewed as a succession of numbers (the ASCII or UTF8
       representation of its elements).
    2. The string is then decomposed into a set of n-grams, i.e.
       n-dimensional vectors of integers.
    3. A hashing function is used to assign an integer to each n-gram.
       The minimum of the hashes over all n-grams is used in the encoding.
    4. This process is repeated with `N` hashing functions to form
       N-dimensional encodings.

    Maxhash encodings can be computed similarly by taking the maximum hash
    instead.
    With this procedure, strings that share many n-grams have a greater
    probability of having the same encoding value. These encodings thus capture
    morphological similarities between strings.

    Parameters
    ----------
    n_components : int, default=30
        The number of dimension of encoded strings. Numbers around 300 tend to
        lead to good prediction performance, but with more computational cost.
    ngram_range : 2-tuple of int, default=(2, 4)
        The lower and upper boundaries of the range of n-values for different
        n-grams used in the string similarity. All values of `n` such
        that ``min_n <= n <= max_n`` will be used.
    hashing : {'fast', 'murmur'}, default='fast'
        Hashing function. `fast` is faster than `murmur` but
        might have some concern with its entropy.
    minmax_hash : bool, default=False
        If `True`, returns the min and max hashes concatenated.
    handle_missing : {'error', 'zero_impute'}, default='zero_impute'
        Whether to raise an error or encode missing values (NaN) with
        vectors filled with zeros.
    n_jobs : int, optional
        The number of jobs to run in parallel.
        The hash computations for all unique elements are parallelized.
        `None` means 1 unless in a joblib.parallel_backend.
        -1 means using all processors.
        See :term:`n_jobs` for more details.

    Attributes
    ----------
    hash_dict_ : LRUDict
        Computed hashes.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    feature_names_in_ : ndarray of shape (n_features_in,)
        Names of features seen during :term:`fit`.

    See Also
    --------
    GapEncoder
        Encodes dirty categories (strings) by constructing latent topics with
        continuous encoding.
    SimilarityEncoder
        Encode string columns as a numeric array with n-gram string similarity.
    deduplicate
        Deduplicate data by hierarchically clustering similar strings.

    References
    ----------
    For a detailed description of the method, see
    `Encoding high-cardinality string categorical variables
    <https://hal.inria.fr/hal-02171256v4>`_ by Cerda, Varoquaux (2019).

    Examples
    --------
    >>> enc = MinHashEncoder(n_components=5)

    Let's encode the following non-normalized data:

    >>> X = [['paris, FR'], ['Paris'], ['London, UK'], ['London']]

    >>> enc.fit(X)
    MinHashEncoder(n_components=5)

    The encoded data with 5 components are:

    >>> enc.transform(X)
    array([[-1.78337518e+09, -1.58827021e+09, -1.66359234e+09,
            -1.81988679e+09, -1.96259387e+09],
           [-8.48046971e+08, -1.76657887e+09, -1.55891205e+09,
            -1.48574446e+09, -1.68729890e+09],
           [-1.97582893e+09, -2.09500033e+09, -1.59652117e+09,
            -1.81759383e+09, -2.09569333e+09],
           [-1.97582893e+09, -2.09500033e+09, -1.53072052e+09,
            -1.45918266e+09, -1.58098831e+09]])
    """

    hash_dict_: LRUDict

    _capacity: int = 2**10

    def __init__(
        self,
        *,
        n_components: int = 30,
        ngram_range: tuple[int, int] = (2, 4),
        hashing: Literal["fast", "murmur"] = "fast",
        minmax_hash: bool = False,
        handle_missing: Literal["error", "zero_impute"] = "zero_impute",
        n_jobs: int = None,
    ):
        self.ngram_range = ngram_range
        self.n_components = n_components
        self.hashing = hashing
        self.minmax_hash = minmax_hash
        self.handle_missing = handle_missing
        self.n_jobs = n_jobs

    def _get_murmur_hash(self, string: str) -> NDArray:
        """
        Encode a string using murmur hashing function.

        Parameters
        ----------
        string : str
            The string to encode.

        Returns
        -------
        ndarray of shape (n_components, )
            The encoded string.
        """
        min_hashes = np.ones(self.n_components) * np.infty
        grams = get_unique_ngrams(string, self.ngram_range)
        if len(grams) == 0:
            grams = get_unique_ngrams(" Na ", self.ngram_range)
        for gram in grams:
            hash_array = np.array(
                [
                    murmurhash3_32("".join(gram), seed=d, positive=True)
                    for d in range(self.n_components)
                ]
            )
            min_hashes = np.minimum(min_hashes, hash_array)
        return min_hashes / (2**32 - 1)

    def _get_fast_hash(self, string: str) -> NDArray:
        """Encode a string with fast hashing function.

        Fast hashing supports both min_hash and minmax_hash encoding.

        Parameters
        ----------
        string : str
            The string to encode.

        Returns
        -------
        ndarray of shape (n_components, )
            The encoded string, using specified encoding scheme.
        """
        if self.minmax_hash:
            return np.concatenate(
                [
                    ngram_min_hash(string, self.ngram_range, seed, return_minmax=True)
                    for seed in range(self.n_components // 2)
                ]
            )
        else:
            return np.array(
                [
                    ngram_min_hash(string, self.ngram_range, seed)
                    for seed in range(self.n_components)
                ]
            )

    def _compute_hash_batched(
        self, batch: Collection[str], hash_func: Callable[[str], NDArray]
    ) -> NDArray:
        """Function called to compute the hashes of a batch of strings.

        Check if the string is in the hash dictionary, if not, compute the hash
        using the specified hashing function and add it to the dictionary.

        Parameters
        ----------
        batch : collection of str
            The batch of strings to encode.
        hash_func : callable
            Hashing function to use on the string.

        Returns
        -------
        ndarray of shape (n_samples, n_components)
            The encoded strings, using specified encoding scheme.
        """
        res = np.zeros((len(batch), self.n_components))
        for i, string in enumerate(batch):
            if string not in self.hash_dict_:
                if string == "NAN":  # true if x is a missing value
                    self.hash_dict_[string] = np.zeros(self.n_components)
                else:
                    self.hash_dict_[string] = hash_func(string)
            res[i] = self.hash_dict_[string]
        return res

    def fit(self, X: ArrayLike, y=None) -> "MinHashEncoder":
        """Fit the MinHashEncoder to `X`.

        In practice, just initializes a dictionary
        to store encodings to speed up computation.

        Parameters
        ----------
        X : array-like, shape (n_samples, ) or (n_samples, n_columns)
            The string data to encode. Only here for compatibility.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        MinHashEncoder
            The fitted MinHashEncoder instance (self).
        """
        self._check_feature_names(X, reset=True)
        X = check_input(X)
        self._check_n_features(X, reset=True)

        if self.hashing not in ["fast", "murmur"]:
            raise ValueError(
                f"Got hashing={self.hashing!r}, "
                "but expected any of {'fast', 'murmur'}. "
            )
        if self.handle_missing not in ["error", "zero_impute"]:
            raise ValueError(
                f"Got handle_missing={self.handle_missing!r}, but expected "
                "any of {'error', 'zero_impute'}. "
            )
        self.hash_dict_ = LRUDict(capacity=self._capacity)
        return self

    def transform(self, X: ArrayLike) -> NDArray:
        """
        Transform `X` using specified encoding scheme.

        Parameters
        ----------
        X : array-like, shape (n_samples, ) or (n_samples, n_columns)
            The string data to encode.

        Returns
        -------
        ndarray of shape (n_samples, n_columns * n_components)
            Transformed input.
        """
        check_is_fitted(self, "hash_dict_")
        self._check_feature_names(X, reset=False)
        X = check_input(X)
        self._check_n_features(X, reset=False)
        if self.minmax_hash:
            if self.n_components % 2 != 0:
                raise ValueError(
                    "n_components should be even when using"
                    f"minmax_hash encoding, got {self.n_components}"
                )
        if self.hashing == "murmur":
            if self.minmax_hash:
                raise ValueError(
                    "minmax_hash encoding is not supported"
                    "with the murmur hashing function"
                )
        if self.handle_missing not in ["error", "zero_impute"]:
            raise ValueError(
                "handle_missing should be either "
                f"'error' or 'zero_impute', got {self.handle_missing!r}"
            )

        # Handle missing values
        missing_mask = (
            ~(X == X)  # Find np.nan
            | (X == None)  # noqa: E711 Find None. Note: `X is None` doesn't work.
            | (X == "")  # Find empty strings
        )

        if missing_mask.any():  # contains at least one missing value
            if self.handle_missing == "error":
                raise ValueError(
                    "Found missing values in input data; set "
                    "handle_missing='zero_impute' to encode with missing values"
                )
            elif self.handle_missing == "zero_impute":
                # NANs will be replaced by zeroes in _compute_hash
                X[missing_mask] = "NAN"

        if self.hashing == "fast":
            hash_func = self._get_fast_hash
        elif self.hashing == "murmur":
            hash_func = self._get_murmur_hash
        else:
            raise ValueError(
                "Hashing function should be either 'fast' or 'murmur', "
                f"got {self.hashing!r}"
            )

        # Compute the hashes for unique values
        unique_x, indices_x = np.unique(X, return_inverse=True)
        n_jobs = effective_n_jobs(self.n_jobs)

        # Compute the hashes in parallel on n_jobs batches
        unique_x_trans = Parallel(n_jobs=n_jobs)(
            delayed(self._compute_hash_batched)(
                unique_x[idx_slice],
                hash_func,
            )
            for idx_slice in gen_even_slices(len(unique_x), n_jobs)
        )

        # Match the hashes of the unique value to the original values
        X_out = np.concatenate(unique_x_trans)[indices_x].reshape(
            len(X), X.shape[1] * self.n_components
        )

        return X_out.astype(np.float64)  # The output is an int32 before conversion

    def get_feature_names_out(
        self, input_features: ArrayLike | str | None = None
    ) -> NDArray[np.str_]:
        """Get output feature names for transformation.

        The output feature names look like:
        ``["x0_0", "x0_1", ..., "x0_(n_components - 1)",
        "x1_0", ..., "x1_(n_components - 1)", ...,
        "x(n_features_out - 1)_(n_components - 1)"]``

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If ``input_features`` is ``None``, then ``feature_names_in_`` is
              used as feature names in. If ``feature_names_in_`` is not defined,
              then the following input feature names are generated:
              ``["x0", "x1", ..., "x(n_features_in_ - 1)"]``.
            - If ``input_features`` is an array-like, then ``input_features`` must
              match ``feature_names_in_`` if ``feature_names_in_`` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """

        check_is_fitted(self)
        input_features = _check_feature_names_in(self, input_features)

        feature_names = []
        for feature in input_features:
            for i in range(self.n_components):
                feature_names.append(f"{feature}_{i}")

        return np.asarray(feature_names, dtype=object)

    def _more_tags(self):
        """
        Used internally by sklearn to ease the estimator checks.
        """
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
            "univariate": True,  # whether the estimator is univariate and can be
            # applied column by column. This is useful for the TableVectorizer,
            # to decide whether to apply the transformer on each column separately
            # and thus improve the parallelization when the transformer is slow enough.
        }
