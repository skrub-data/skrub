import collections
import numpy as np
from scipy import sparse

from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.feature_extraction.text import CountVectorizer

import jellyfish
import Levenshtein as lev
import distance as dist

from .ngrams_vectorizer import ngram_similarity

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoding='onehot', similarity='ngram',
                 ngram_type='sim2',
                 n=3, categories='auto',
                 dtype=np.float64, handle_unknown='error',
                 clf_type='binary_clf'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.clf_type = clf_type
        self.handle_unknown = handle_unknown
        self.similarity = similarity
        self.ngram_type = ngram_type
        self.n = n

    def fit(self, X, y=None):
        if self.encoding not in ['similarity',
                                 'target',
                                 'ordinal',
                                 'onehot',
                                 'onehot-dense',
                                 'ngram-count',
                                 'ngram-presence',
                                 'ngram-tfidf']:
            template = ("Encoding %s has not been implemented yet")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

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
        if self.encoding == 'target':
            self.Eyx_ = [{cat: np.mean(y[X[:, i] == cat])
                          for cat in self.categories_[i]}
                         for i in range(len(self.categories_))]
            self.Ey_ = [np.mean(y)
                        for i in range(len(self.categories_))]
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

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        if self.encoding == 'ngram-count':
            out = []
            for j, cats in enumerate(self.categories_):
                n = int(encoder[0])
                vectorizer = CountVectorizer(analyzer='char',
                                             ngram_range=(self.n, self.n))
                vectorizer.fit(cats)
                encoder = vectorizer.transform(X[:, j])
                out.append(encoder)
            return sparse.hstack(out)

        if self.encoding == 'ngram-presence':
            out = []
            for j, cats in enumerate(self.categories_):
                vectorizer = CountVectorizer(analyzer='char',
                                             ngram_range=(self.n, self.n))
                vectorizer.fit(cats)
                encoder = vectorizer.transform(X[:, j])
                encoder = (encoder > 0).astype(self.dtype)
                out.append(encoder)
            return sparse.hstack(out)

        if self.encoding == 'ngram-tfidf':
            out = []
            for j, cats in enumerate(self.categories_):
                n = int(encoder[0])
                B = np.unique(B)
                vectorizer = TfidfVectorizer(analyzer='char',
                                             ngram_range=(self.n, self.n),
                                             smooth_idf=False)
                vectorizer.fit(cats)
                encoder = vectorizer.transform(X[:, j])
                out.append(encoder)
            return np.hstack(out)

        if self.encoding == 'similarity':
            if self.similarity == 'levenshtein-ratio':
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

            if self.similarity == 'sorensen':
                out = []
                for j, cats in enumerate(self.categories_):
                    unqX = np.unique(X[:, j])
                    vect = np.vectorize(dist.sorensen)
                    encoder_dict = {x: vect(x, cats.reshape(1, -1))
                                    for x in unqX}
                    encoder = [encoder_dict[x] for x in X[:, j]]
                    encoder = 1 - np.vstack(encoder)
                    out.append(encoder)
                return np.hstack(out)

            if self.similarity == 'jaro-winkler':
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

            if self.similarity == 'ngram':
                out = []
                for j, cats in enumerate(self.categories_):
                    encoder = ngram_similarity(X[:, j], cats,
                                               self.n, self.ngram_type,
                                               dtype=self.dtype)
                    out.append(encoder)
                # '3gram_similarity2',
                # '3gram_similarity2_1',
                # '3gram_similarity4',
                # '3gram_similarity2_2',
                # '3gram_similarity5',
                return np.hstack(out)
        if self.encoding == 'target':
            def lambda_(x, n):
                out = x / (x + n)
                # out = 1.0
                return 1.0
            out = []
            for j, cats in enumerate(self.categories_):
                counter = collections.Counter(X[:, j])
                unqX = np.unique(X[:, j])
                n = len(X[:, j])
                k = len(cats)
                encoder = {x: 0 for x in unqX}
                if self.clf_type in ['binary_clf', 'regression']:
                    for x in unqX:
                        if x not in cats:
                            Eyx = 0
                        else:
                            Eyx = self.Eyx_[j][x]
                        lambda_n = lambda_(counter[x], n/k)
                        encoder[x] = lambda_n*Eyx + \
                            (1 - lambda_n)*self.Ey_[j]
                    x_out = np.zeros((len(X[:, j]), 1))
                    for i, x in enumerate(X[:, j]):
                        x_out[i, 0] = encoder[x]
                    out.append(x_out.reshape(-1, 1))
            out = np.hstack(out)
            return out

        if self.encoding == 'onehot':
            encoder = []
            for j, cats in enumerate(self.categories_):
                unqX = np.unique(X[:, j])
                cats_dict = {s: i for i, s in enumerate(cats)}
                encoder_unq = sparse.lil_matrix((len(unqX), len(cats)))
                for i, s in enumerate(unqX):
                    try:
                        encoder_unq[i, cats_dict[s]] = 1
                    except KeyError:
                        continue
                unqX_dict = {s: i for i, s in enumerate(unqX)}
                index = [unqX_dict[s] for s in X[:, j]]
                encoder.append(encoder_unq[index])
            out = sparse.hstack(encoder)
            return sparse.csr_matrix(out)
        if self.encoding == 'onehot-dense':
            encoder = []
            for j, cats in enumerate(self.categories_):
                unqX = np.unique(X[:, j])
                cats_dict = {s: i for i, s in enumerate(cats)}
                encoder_unq = sparse.lil_matrix((len(unqX), len(cats)))
                for i, s in enumerate(unqX):
                    try:
                        encoder_unq[i, cats_dict[s]] = 1
                    except KeyError:
                        continue
                unqX_dict = {s: i for i, s in enumerate(unqX)}
                index = [unqX_dict[s] for s in X[:, j]]
                encoder.append(encoder_unq[index])
            out = sparse.hstack(encoder)
            return out.toarray()
        else:
            return out

    def inverse_transform(self, X):
        """Convert back the data to the original representation.
        In case unknown categories are encountered (all zero's in the
        one-hot encoding), ``None`` is used to represent this category.
        Parameters
        ----------
        X : array-like or sparse matrix, shape [n_samples, n_encoded_features]
            The transformed data.
        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Inverse transformed array.
        """
        check_is_fitted(self, 'categories_')
        X = check_array(X, accept_sparse='csr')

        n_samples, _ = X.shape
        n_features = len(self.categories_)
        n_transformed_features = sum([len(cats) for cats in self.categories_])

        # validate shape of passed X
        msg = ("Shape of the passed X data is not correct. Expected {0} "
               "columns, got {1}.")
        if self.encoding == 'ordinal' and X.shape[1] != n_features:
            raise ValueError(msg.format(n_features, X.shape[1]))
        elif (self.encoding.startswith('onehot')
                and X.shape[1] != n_transformed_features):
            raise ValueError(msg.format(n_transformed_features, X.shape[1]))

        # create resulting array of appropriate dtype
        dt = np.find_common_type([cat.dtype for cat in self.categories_], [])
        X_tr = np.empty((n_samples, n_features), dtype=dt)

        if self.encoding == 'ordinal':
            for i in range(n_features):
                labels = X[:, i].astype('int64')
                X_tr[:, i] = self.categories_[i][labels]

        else:  # encoding == 'onehot' / 'onehot-dense'
            j = 0
            found_unknown = {}

            for i in range(n_features):
                n_categories = len(self.categories_[i])
                sub = X[:, j:j + n_categories]

                # for sparse X argmax returns 2D matrix, ensure 1D array
                labels = np.asarray(_argmax(sub, axis=1)).flatten()
                X_tr[:, i] = self.categories_[i][labels]

                if self.handle_unknown == 'ignore':
                    # ignored unknown categories: we have a row of all zero's
                    unknown = np.asarray(sub.sum(axis=1) == 0).flatten()
                    if unknown.any():
                        found_unknown[i] = unknown

                j += n_categories

            # if ignored are found: potentially need to upcast result to
            # insert None values
            if found_unknown:
                if X_tr.dtype != object:
                    X_tr = X_tr.astype(object)

                for idx, mask in found_unknown.items():
                    X_tr[mask, idx] = None

        return X_tr
