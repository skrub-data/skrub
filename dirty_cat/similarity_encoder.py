import numpy as np
from scipy import sparse

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

import jellyfish
import Levenshtein as lev
import distance as dist





def ngram_similarity(X, cats, n, sim_type=None, dtype=np.float64):
    """ Similarity encoding for dirty categorical variables:
    Given to arrays of strings, returns the
    similarity encoding matrix of size
    len(X) x len(cats)

    sim1(s_i, s_j) = 2||min(ci, cj)||_1/ (||ci||_1 + ||cj||_1)

    sim2(s_i, s_j) = 2 dot(c1, c2) / (dot(c1, c1) + dot(c2, c2))

    sim3(s_i, s_j) = dot(c1, c2) / (dot(c1, c1)^.5 * dot(c2, c2)^.5)

    sim4(s_i, s_j) = 2 dot(p1, p2) / (dot(p1, p1) + dot(p2, p2))

    sim5(s_i, s_j) = dot(p1, p2) / (dot(p1, p1)^.5 * dot(p2, p2)^.5)

    """

    def sim1():
        """
        sim1(s_i, s_j) = 2||min(ci, cj)||_1/ (||ci||_1 + ||cj||_1)
        """
        unq_X = np.unique(X)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
        count2 = vectorizer.fit_transform(cats)
        count1 = vectorizer.transform(unq_X)
        sum_matrix2 = count2.sum(axis=1)
        SE_dict = {}
        for i, x in enumerate(count1):
            aux = sparse.csr_matrix(np.ones((count2.shape[0], 1))).dot(x)
            samegrams = count2.minimum(aux).sum(axis=1)
            allgrams = x.sum() + sum_matrix2
            similarity = 2 * np.divide(samegrams, allgrams)
            SE_dict[unq_X[i]] = np.array(similarity).reshape(-1)
        SE = []
        for x in X:
            SE.append(SE_dict[x])
        return np.nan_to_num(np.vstack(SE))

    def sim2():
        """
        sim2(s_i, s_j) = 2 dot(c1, c2) / (dot(c1, c1) + dot(c2, c2)
        """
        unq_X = np.unique(X)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
        Cj = vectorizer.fit_transform(cats).transpose()
        Ci = vectorizer.transform(unq_X)
        SE_dict = {}
        cij = Ci.dot(Cj).toarray()
        cii = np.tile(Ci.multiply(Ci).sum(axis=1),
                      (1, Cj.shape[1]))
        cjj = np.tile(Cj.multiply(Cj).sum(axis=0),
                      (Ci.shape[0], 1))
        similarity = np.divide(2*cij, cii + cjj)
        X_dict = {s: i for i, s in enumerate(unq_X)}
        index = [X_dict[x] for x in X]
        similarity = similarity[index]
        return np.nan_to_num(similarity)

    def sim3():
        """
        sim3(s_i, s_j) = dot(c1, c2) / (dot(c1, c1)^.5 * dot(c2, c2)^.5)
        """
        unq_X = np.unique(X)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
        Cj = vectorizer.fit_transform(cats).transpose()
        Ci = vectorizer.transform(unq_X)
        SE_dict = {}
        cij = Ci.dot(Cj).toarray()
        cii = np.tile(np.power(Ci.multiply(Ci).sum(axis=1), .5),
                      (1, Cj.shape[1]))
        cjj = np.tile(np.power(Cj.multiply(Cj).sum(axis=0), .5),
                      (Ci.shape[0], 1))
        similarity = np.divide(cij, np.multiply(cii, cjj))
        X_dict = {s: i for i, s in enumerate(unq_X)}
        index = [X_dict[x] for x in X]
        similarity = similarity[index]
        return np.nan_to_num(similarity)

    def sim4():
        """
        sim4(s_i, s_j) = 2 dot(p1, p2) / (dot(p1, p1) + dot(p2, p2))
        """
        unq_X = np.unique(X)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
        Cj = (vectorizer.fit_transform(cats) > 0
              ).astype(dtype).transpose()
        Ci = (vectorizer.transform(unq_X) > 0).astype(dtype)
        SE_dict = {}
        cij = Ci.dot(Cj).toarray()
        cii = np.tile(Ci.multiply(Ci).sum(axis=1),
                      (1, Cj.shape[1]))
        cjj = np.tile(Cj.multiply(Cj).sum(axis=0),
                      (Ci.shape[0], 1))
        similarity = np.divide(2*cij, cii + cjj)
        X_dict = {s: i for i, s in enumerate(unq_X)}
        index = [X_dict[x] for x in X]
        similarity = similarity[index]
        return np.nan_to_num(similarity)

    def sim5():
        """
        sim5(s_i, s_j) = dot(p1, p2) / (dot(p1, p1)^.5 * dot(p2, p2)^.5)
        """
        unq_X = np.unique(X)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
        Cj = (vectorizer.fit_transform(cats) > 0
              ).astype(dtype).transpose()
        Ci = (vectorizer.transform(unq_X) > 0).astype(dtype)
        SE_dict = {}
        cij = Ci.dot(Cj).toarray()
        cii = np.tile(np.power(Ci.multiply(Ci).sum(axis=1), .5),
                      (1, Cj.shape[1]))
        cjj = np.tile(np.power(Cj.multiply(Cj).sum(axis=0), .5),
                      (Ci.shape[0], 1))
        similarity = np.divide(cij, np.multiply(cii, cjj))
        X_dict = {s: i for i, s in enumerate(unq_X)}
        index = [X_dict[x] for x in X]
        similarity = similarity[index]
        return np.nan_to_num(similarity)

    if sim_type == 'sim1':
        return sim1()

    if sim_type == 'sim2':
        return sim2()

    if sim_type == 'sim3':
        return sim3()

    if sim_type == 'sim4':
        return sim4()

    if sim_type == 'sim5':
        return sim4()



class SimilarityEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, similarity='ngram',
                 ngram_similarity_type='sim2',
                 n=3, categories='auto',
                 dtype=np.float64, handle_unknown='ignore',
                 clf_type='binary_clf', ngram_type=None):
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.similarity = similarity
        self.ngram_type = ngram_type
        self.n = n

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
            return np.hstack(out)
