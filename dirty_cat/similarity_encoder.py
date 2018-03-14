import numpy as np
from scipy import sparse

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

import jellyfish
import Levenshtein as lev


def ngram_similarity(X, cats, n_min, n_max, sim_type=None, dtype=np.float64):
    """
    Similarity encoding for dirty categorical variables:
        Given to arrays of strings, returns the
        similarity encoding matrix of size
        len(X) x len(cats)

    ngram_sim(s_i, s_j) =
        ||min(ci, cj)||_1 / (||ci||_1 + ||cj||_1 - ||min(ci, cj)||_1)
    """
    unq_X = np.unique(X)
    cats = np.array(['  %s  ' % cat for cat in cats])
    unq_X_ = np.array(['  %s  ' % x for x in unq_X])
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(n_min, n_max))
    vectorizer.fit(np.concatenate((cats, unq_X_)))
    count2 = vectorizer.transform(cats)
    count1 = vectorizer.transform(unq_X_)
    sum2 = count2.sum(axis=1)
    SE_dict = {}
    for i, x in enumerate(count1):
        aux = sparse.csr_matrix(np.ones((count2.shape[0], 1))).dot(x)
        samegrams = count2.minimum(aux).sum(axis=1)
        allgrams = x.sum() + sum2 - samegrams
        similarity = np.divide(samegrams, allgrams)
        SE_dict[unq_X[i]] = np.array(similarity).reshape(-1)
    out = []
    for x in X:
        out.append(SE_dict[x])
    return np.nan_to_num(np.vstack(out))


class SimilarityEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, similarity_type='ngram',
                 n_min=3, n_max=3, categories='auto',
                 dtype=np.float64, handle_unknown='ignore'):
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.similarity_type = similarity_type
        self.n_min = n_min
        self.n_max = n_max

    def fit(self, X, y=None):
        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.categories != 'auto':
            for cats in self.categories:
                if not np.all(np.sort(cats) == np.array(cats)):
                    raise ValueError("Unsorted categories are not yet "
                                     "supported")

        X_temp = check_array(X, dtype=None)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                if self.handle_unknown == 'error':
                    valid_mask = np.in1d(Xi, self.categories[i])
                    if not np.all(valid_mask):
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(self.categories[i])

        self.categories_ = [le.classes_ for le in self._label_encoders_]
        return self

    def transform(self, X):
        """Transform X using specified encoding scheme.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X_temp = check_array(X, dtype=None)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            Xi = X[:, i]
            valid_mask = np.in1d(Xi, self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    Xi = Xi.copy()
                    Xi[~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(Xi)

        if self.similarity_type == 'levenshtein-ratio':
            out = []
            for j, cats in enumerate(self.categories_):
                unqX = np.unique(X[:, j])
                vect = np.vectorize(lev.ratio)
                encoder_dict = {x: vect(x, cats.reshape(1, -1))
                                for x in unqX}
                encoder = [encoder_dict[x] for x in X[:, j]]
                encoder = np.vstack(encoder)
                out.append(encoder)
            return np.hstack(out)

        if self.similarity_type == 'jaro-winkler':
            out = []
            for j, cats in enumerate(self.categories_):
                unqX = np.unique(X[:, j])
                vect = np.vectorize(jellyfish.jaro_distance)
                encoder_dict = {x: vect(x, cats.reshape(1, -1))
                                for x in unqX}
                encoder = [encoder_dict[x] for x in X[:, j]]
                encoder = np.vstack(encoder)
                out.append(encoder)
            return np.hstack(out)

        if self.similarity_type == 'ngram':
            out = []
            for j, cats in enumerate(self.categories_):
                encoder = ngram_similarity(X[:, j], cats,
                                           self.n_min, self.n_max,
                                           dtype=self.dtype)
                out.append(encoder)
            return np.hstack(out)
