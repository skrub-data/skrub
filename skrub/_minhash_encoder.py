"""
Implements the MinHashEncoder, which encodes string categorical features by
applying the MinHash method to n-gram decompositions of strings.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import TransformerMixin
from sklearn.utils import gen_even_slices, murmurhash3_32
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._fast_hash import ngram_min_hash
from ._on_each_column import RejectColumn, SingleColumnTransformer
from ._string_distances import get_unique_ngrams
from ._utils import LRUDict, unique_strings

NoneType = type(None)


class MinHashEncoder(TransformerMixin, SingleColumnTransformer):
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
    >>> import pandas as pd
    >>> from skrub import MinHashEncoder
    >>> enc = MinHashEncoder(n_components=5)

    Let's encode the following non-normalized data:

    >>> X = pd.Series(['paris, FR', 'Paris', 'London, UK', 'London'], name='city')
    >>> enc.fit(X)
    MinHashEncoder(n_components=5)

    The encoded data with 5 components are:

    >>> enc.transform(X)
             city_0        city_1        city_2        city_3        city_4
    0 -1.783375e+09 -1.588270e+09 -1.663592e+09 -1.819887e+09 -1.962594e+09
    1 -8.480470e+08 -1.766579e+09 -1.558912e+09 -1.485745e+09 -1.687299e+09
    2 -1.975829e+09 -2.095000e+09 -1.596521e+09 -1.817594e+09 -2.095693e+09
    3 -1.975829e+09 -2.095000e+09 -1.530721e+09 -1.459183e+09 -1.580988e+09
    """

    _capacity = 2**10

    def __init__(
        self,
        *,
        n_components=30,
        ngram_range=(2, 4),
        hashing="fast",
        minmax_hash=False,
        n_jobs=None,
    ):
        self.ngram_range = ngram_range
        self.n_components = n_components
        self.hashing = hashing
        self.minmax_hash = minmax_hash
        self.n_jobs = n_jobs

    def _get_murmur_hash(self, string):
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
        min_hashes = np.ones(self.n_components) * np.inf
        grams = get_unique_ngrams(string, self.ngram_range)
        if string == "" or len(grams) == 0:
            return np.zeros(self.n_components)
        for gram in grams:
            hash_array = np.array(
                [
                    murmurhash3_32("".join(gram), seed=d, positive=True)
                    for d in range(self.n_components)
                ]
            )
            min_hashes = np.minimum(min_hashes, hash_array)
        return min_hashes / (2**32 - 1)

    def _get_fast_hash(self, string):
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

    def _compute_hash_batched(self, batch, hash_func):
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
                self.hash_dict_[string] = hash_func(string)
            res[i] = self.hash_dict_[string]
        return res

    def fit(self, X, y=None):
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
        if not (sbd.is_categorical(X) or sbd.is_string(X)):
            raise RejectColumn(f"Column {sbd.name(X)!r} does not contain strings.")
        if self.hashing not in ["fast", "murmur"]:
            raise ValueError(
                f"Got hashing={self.hashing!r}, "
                "but expected any of {'fast', 'murmur'}. "
            )
        if self.minmax_hash and self.n_components % 2 != 0:
            raise ValueError(
                "n_components should be even when using"
                f"minmax_hash encoding, got {self.n_components}"
            )
        if self.hashing == "murmur" and self.minmax_hash:
            raise ValueError(
                "minmax_hash encoding is not supported with the murmur hashing function"
            )
        self.hash_dict_ = LRUDict(capacity=self._capacity)
        self._input_name = sbd.name(X)
        return self

    def transform(self, X):
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
        if not (sbd.is_categorical(X) or sbd.is_string(X)):
            raise ValueError(f"Column {sbd.name(X)!r} does not contain strings.")

        X_values = sbd.to_numpy(X)
        if self.hashing == "fast":
            hash_func = self._get_fast_hash
        else:
            # already checked during fit
            assert self.hashing == "murmur", self.hashing
            hash_func = self._get_murmur_hash

        is_null = sbd.to_numpy(sbd.is_null(X))
        unique_x, indices_x = unique_strings(X_values, is_null)
        n_jobs = effective_n_jobs(self.n_jobs)

        # Compute the hashes in parallel on n_jobs batches
        unique_x_trans = Parallel(n_jobs=n_jobs)(
            delayed(self._compute_hash_batched)(
                unique_x[idx_slice],
                hash_func,
            )
            for idx_slice in gen_even_slices(len(unique_x), n_jobs)
        )
        X_out = np.concatenate(unique_x_trans, dtype="float32")[indices_x]
        names = self.get_feature_names_out()
        result = sbd.make_dataframe_like(X, dict(zip(names, X_out.T)))
        result = sbd.copy_index(X, result)
        return result

    def get_feature_names_out(self):
        """Get output feature names for transformation.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """

        check_is_fitted(self)
        return [f"{self._input_name}_{i}" for i in range(self.n_components)]
