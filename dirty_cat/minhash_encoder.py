"""
Minhash encoding of string arrays.
The principle is as follows:
  1. A string is viewed as a succession of numbers (the ASCII or UTF8
     representation of its elements).
  2. The string is then decomposed into a set of n-grams, i.e.
     n-dimensional vectors of integers.
  3. A hashing function is used to assign an integer to each n-gram.
     The minimum of the hashes over all n-grams is used in the encoding.
  4. This process is repeated with N hashing functions are used to 
     form N-dimensional encodings.
Maxhash encodings can be computed similarly by taking the hashes maximum
instead.
With this procedure, strings that share many n-grams have greater
probability of having same encoding values. These encodings thus capture
morphological similarities between strings.
"""
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, murmurhash3_32

from .fast_hash import ngram_min_hash
from .utils import LRUDict


class MinHashEncoder(BaseEstimator, TransformerMixin):
    """
    Encode string categorical features as a numeric array, minhash method
    applied to ngram decomposition of strings based on ngram decomposition
    of the string.

    Parameters
    ----------
    n_components : int
        The number of dimension of encoded strings.
    ngram_range : tuple (min_n, max_n), default=(2, 4)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n.
        will be used.
    hashing : str {'fast', 'murmur'}, default=fast
        Hashing function. fast is faster but
        might have some concern with its entropy.
    minmax_hash : bool, default=False
        if True, return min hash and max hash concatenated.
        handle_missing : 'error' or '' (default)
        Whether to raise an error or impute with blank string '' if missing
        values (NaN) are present during fit (default is to impute).
        When this parameter is set to '', and a missing value is encountered
        during fit_transform, the resulting encoded columns for this feature
        will be all zeros.
    References
    ----------
    For a detailed description of the method, see
    `Encoding high-cardinality string categorical variables
    <https://hal.inria.fr/hal-02171256v4>`_ by Cerda, Varoquaux (2019).

    """

    def __init__(self, n_components, ngram_range=(2, 4),
                 hashing='fast', minmax_hash=False,
                 handle_missing=''):
        self.ngram_range = ngram_range
        self.n_components = n_components
        self.hashing = hashing
        self.minmax_hash = minmax_hash
        self.count = 0
        self.handle_missing = handle_missing

    def get_unique_ngrams(self, string, ngram_range):
        """ Return the set of unique n-grams of a string.
        Parameters
        ----------
        string : str
            The string to split in n-grams.
        ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n.
        Returns
        -------
        set
            The set of unique n-grams of the string.
        """
        spaces = ' '  # * (n // 2 + n % 2)
        string = spaces + " ".join(string.lower().split()) + spaces
        ngram_set = set()
        for n in range(ngram_range[0], ngram_range[1] + 1):
            string_list = [string[i:] for i in range(n)]
            ngram_set |= set(zip(*string_list))
        return ngram_set

    def minhash(self, string, n_components, ngram_range):
        """ Encode a string using murmur hashing function.
        Parameters
        ----------
        string : str
            The string to encode.
        n_components : int
            The number of dimension of encoded string.
        ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n.
        Returns
        -------
        array, shape (n_components, )
            The encoded string.
        """
        min_hashes = np.ones(n_components) * np.infty
        grams = self.get_unique_ngrams(string, self.ngram_range)
        if len(grams) == 0:
            grams = self.get_unique_ngrams(' Na ', self.ngram_range)
        for gram in grams:
            hash_array = np.array([
                murmurhash3_32(''.join(gram), seed=d, positive=True)
                for d in range(n_components)])
            min_hashes = np.minimum(min_hashes, hash_array)
        return min_hashes/(2**32-1)

    def get_fast_hash(self, string):
        """
        Encode a string with fast hashing function.
        fast hashing supports both min_hash and minmax_hash encoding.
        Parameters
        ----------
        string : str
            The string to encode.
        Returns
        -------
        array, shape (n_components, )
            The encoded string, using specified encoding scheme.
        """
        if self.minmax_hash:
            return np.concatenate([ngram_min_hash(string, self.ngram_range,
                                   seed, return_minmax=True)
                                   for seed in range(self.n_components//2)])
        else:
            return np.array([ngram_min_hash(string, self.ngram_range, seed)
                            for seed in range(self.n_components)])

    def get_murmur_hash(self, string):
        """
        Encode a string with murmur hashing function.
        murmur only supports min_hash encoding.
        Parameters
        ----------
        string : str
            The string to encode.
        Returns
        -------
        array, shape (n_components, )
            The encoded string, using specified encoding scheme.
        """
        return self.minhash(
                    string, n_components=self.n_components,
                    ngram_range=self.ngram_range)

    def fit(self, X, y=None):
        """
        Fit the MinHashEncoder to X. In practice, just initializes a dictionary
        to store encodings to speed up computation.
        Parameters
        ----------
        X : array-like, shape (n_samples, ) or (n_samples, 1)
            The string data to encode.

        Returns
        -------
        self
            The fitted MinHashEncoder instance.
        """
        self.hash_dict = LRUDict(capacity=2**10)
        return self

    def transform(self, X):
        """ Transform X using specified encoding scheme.
        Parameters
        ----------
        X : array-like, shape (n_samples, ) or (n_samples, 1)
            The string data to encode.
        Returns
        -------
        array, shape (n_samples, n_components)
            Transformed input.
        """
        X = np.asarray(X)
        assert X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1), f"ERROR:\
        shape {X.shape} of input array is not supported."
        if X.ndim == 2:
            X = X[:, 0]
        # Check if first item has str or np.str_ type
        assert isinstance(X[0], str), "ERROR: Input data is not string."
        if self.minmax_hash:
            assert self.n_components % 2 == 0,\
                    "n_components should be even when minmax_hash=True"
        if self.hashing == 'murmur':
            assert not(self.minmax_hash),\
                   "minmax_hash not implemented with murmur"
        if self.handle_missing not in ['error', '']:
            template = ("handle_missing should be either 'error' or "
                        "'', got %s")
            raise ValueError(template % self.handle_missing)
        X_out = np.zeros((len(X), self.n_components))
        X = X.ravel()

        # TODO Parallel run here
        nan_idx = []

        if self.hashing == 'fast':
            for i, x in enumerate(X):
                if isinstance(x, float):
                    nan_idx.append(i)
                elif x not in self.hash_dict:
                    self.hash_dict[x] = self.get_fast_hash(x)

        elif self.hashing == 'murmur':
            for i, x in enumerate(X):
                if isinstance(x, float):
                    nan_idx.append(i)
                elif x not in self.hash_dict:
                    self.hash_dict[x] = self.get_murmur_hash(x)

        else:
            raise ValueError("hashing function must be 'fast' or"
                             "'murmur', got '{}'"
                             "".format(self.hashing))

        if self.handle_missing == 'error' and nan_idx:
            msg = ("Found missing values in input data; set "
                   "handle_missing='' to encode with missing values")
            raise ValueError(msg)

        for i, x in enumerate(X):
            if i not in nan_idx:
                X_out[i, :] = self.hash_dict[x]

        return X_out
