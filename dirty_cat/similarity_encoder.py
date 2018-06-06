import numpy as np
from scipy import sparse

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


from dirty_cat import string_distances


def ngram_similarity(X, cats, ngram_range, dtype=np.float64):
    """
    Similarity encoding for dirty categorical variables:
        Given to arrays of strings, returns the
        similarity encoding matrix of size
        len(X) x len(cats)

    ngram_sim(s_i, s_j) =
        ||min(ci, cj)||_1 / (||ci||_1 + ||cj||_1 - ||min(ci, cj)||_1)
    """
    min_n, max_n = ngram_range
    unq_X = np.unique(X)
    cats = np.array([' %s ' % cat for cat in cats])
    unq_X_ = np.array([' %s ' % x for x in unq_X])
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(min_n, max_n))
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


_VECTORIZED_EDIT_DISTANCES = {
    'levenshtein-ratio': np.vectorize(string_distances.levenshtein_ratio),
    'jaro': np.vectorize(string_distances.jaro),
    'jaro-winkler': np.vectorize(string_distances.jaro_winkler),
}


class SimilarityEncoder(BaseEstimator, TransformerMixin):
    """Encode string categorical features as a numeric array.

    The input to this transformer should be an array-like of
    strings.
    The method is based on calculating the morphological similarities
    between the categories.
    The categories can be encoded using one of the implemented string
    similarities: ``similarity='ngram'`` (default), 'levenshtein-ratio',
    'jaro', or 'jaro-winkler'.
    This encoding is an alternative to OneHotEncoder in the case of
    dirty categorical variables.

    Parameters
    ----------
    similarity : str {'ngram', 'levenshtein-ratio', 'jaro', or\
'jaro-winkler'}
        The type of pairwise string similarity to use.

    ngram_range : tuple (min_n, max_n), default=(3, 3)
        Only significant for ``similarity='ngram'``. The range of
        values for the n_gram similarity.

    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories must be sorted and should not mix
          strings and numeric values.

        The categories used can be found in the ``categories_`` attribute.
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros. In the inverse transform, an unknown category
        will be denoted as None.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order corresponding with output of ``transform``).

    References
    ----------

    For a detailed description of the method, see
    `Similarity encoding for learning with dirty categorical variables
    <https://hal.inria.fr/hal-01806175>`_ by Cerda, Varoquaux, Kegl. 2018
    (accepted for publication at: Machine Learning journal, Springer).


    """

    def __init__(self, similarity='ngram',
                 ngram_range=(3, 3), categories='auto',
                 dtype=np.float64, handle_unknown='ignore'):
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.similarity = similarity
        self.ngram_range = ngram_range

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

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
        X_new : 2-d array, shape [n_samples, n_features_new]
            Transformed input.

        """
        X_temp = check_array(X, dtype=None)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        n_samples, n_features = X.shape

        for i in range(n_features):
            Xi = X[:, i]
            valid_mask = np.in1d(Xi, self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)

        if self.similarity in ('levenshtein-ratio',
                               'jaro',
                               'jaro-winkler'):
            out = []
            vect = _VECTORIZED_EDIT_DISTANCES[self.similarity]
            for j, cats in enumerate(self.categories_):
                unqX = np.unique(X[:, j])
                encoder_dict = {x: vect(x, cats.reshape(1, -1))
                                for x in unqX}
                encoder = [encoder_dict[x] for x in X[:, j]]
                encoder = np.vstack(encoder)
                out.append(encoder)
            return np.hstack(out)

        elif self.similarity == 'ngram':
            min_n, max_n = self.ngram_range
            out = []
            for j, cats in enumerate(self.categories_):
                encoder = ngram_similarity(X[:, j], cats,
                                           ngram_range=(min_n, max_n),
                                           dtype=self.dtype)
                out.append(encoder)
            return np.hstack(out)
        else:
            raise ValueError("Unknown similarity: '%s'" % self.similarity)
