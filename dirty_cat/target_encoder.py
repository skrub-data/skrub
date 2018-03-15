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

        if self.clf_type in ['binary-clf', 'regression']:
            out = []
            for j, cats in enumerate(self.categories_):
                counter = collections.Counter(X[:, j])
                unqX = np.unique(X[:, j])
                n = len(X[:, j])
                k = len(cats)
                encoder = {x: 0 for x in unqX}
                if self.clf_type in ['binary-clf', 'regression']:
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
        # if self.clf_type == 'multiclass-clf':
        #     x_out = np.zeros((len(cats), len(np.unique(y_train))))
        #     lambda_n = {x: 0 for x in unqA}
        #     y_train2 = {x: 0 for x in unqA}
        #     for x in unqA:
        #         lambda_n[x] = lambda_(counter[x], n/k)
        #         y_train2[x] = y_train[B == x]
        #     for j, y in enumerate(np.unique(y_train)):
        #         Ey = sum(y_train == y)/n
        #         for x in unqA:
        #             if len(y_train2[x]) == 0:
        #                 Eyx = 0
        #             else:
        #                 Eyx = sum(y_train2[x] == y)/len(y_train2[x])
        #             encoder[x] = lambda_n[x]*Eyx + (1 - lambda_n[x])*Ey
        #         for i, x in enumerate(A):
        #             x_out[i, j] = encoder[x]
        #
        #     return out
