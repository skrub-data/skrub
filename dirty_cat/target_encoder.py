import collections
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


def lambda_(x, n):
    out = x / (x + n)
    return out


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 categories='auto',
                 clf_type='binary-clf',
                 dtype=np.float64, handle_unknown='error',
                 ):
        self.categories = categories
        self.dtype = dtype
        self.clf_type = clf_type
        self.handle_unknown = handle_unknown

    def fit(self, X, y):
        """Fit the TargetEncoder to X.
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

        for j in range(n_features):
            le = self._label_encoders_[j]
            Xj = X[:, j]
            if self.categories == 'auto':
                le.fit(Xj)
            else:
                if self.handle_unknown == 'error':
                    valid_mask = np.in1d(Xj, self.categories[j])
                    if not np.all(valid_mask):
                        diff = np.unique(Xj[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, j))
                        raise ValueError(msg)
                le.classes_ = np.array(self.categories[j])

        self.categories_ = [le.classes_ for le in self._label_encoders_]
        self.classes_ = np.unique(y)

        self.Eyx_ = {c: [{cat: np.mean((y == c)[X[:, j] == cat])
                          for cat in self.categories_[j]}
                         for j in range(len(self.categories_))]
                     for c in self.classes_}
        self.Ey_ = {c: np.mean(y == c) for c in self.classes_}
        self.counter_ = {j: collections.Counter(X[:, j])
                         for j in range(n_features)}
        return self

    def transform(self, X):
        """Transform X using specified encoding scheme.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.

        Returns
        -------
        X_out : 2-d array
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

        out = []
        for j, cats in enumerate(self.categories_):
            unqX = np.unique(X[:, j])
            n = len(X[:, j])
            k = len(cats)
            encoder = {x: 0 for x in unqX}
            if self.clf_type in ['binary-clf', 'regression']:
                for x in unqX:
                    if x not in cats:
                        Eyx = 0
                    else:
                        Eyx = self.Eyx_[self.classes_[1]][j][x]
                    lambda_n = lambda_(self.counter_[j][x], n/k)
                    encoder[x] = lambda_n*Eyx + \
                        (1 - lambda_n)*self.Ey_[self.classes_[1]]
                x_out = np.zeros((len(X[:, j]), 1))
                for i, x in enumerate(X[:, j]):
                    x_out[i, 0] = encoder[x]
                out.append(x_out.reshape(-1, 1))
            if self.clf_type == 'multiclass-clf':
                x_out = np.zeros((len(X[:, j]), len(self.classes_)))
                lambda_n = {x: 0 for x in unqX}
                for x in unqX:
                    lambda_n[x] = lambda_(self.counter_[j][x], n/k)
                for k, c in enumerate(np.unique(self.classes_)):
                    for x in unqX:
                        if x not in cats:
                            Eyx = 0
                        else:
                            Eyx = self.Eyx_[c][j][x]
                        encoder[x] = lambda_n[x]*Eyx + \
                            (1 - lambda_n[x])*self.Ey_[c]
                    for i, x in enumerate(X[:, j]):
                        x_out[i, k] = encoder[x]
                out.append(x_out)
        out = np.hstack(out)
        return out
