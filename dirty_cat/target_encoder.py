import collections
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.fixes import _object_dtype_isnan
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted
from dirty_cat.utils import check_input


def lambda_(x, n):
    out = x / (x + n)
    return out


def arr_mean(X_g, y_g):
    """ Find the group mean of numpy arrays with categories """
    X_uniques = np.unique(X_g)
    grouped = [y_g[X_g == xi] for xi in X_uniques]
    mean = [gr.mean() for gr in grouped]
    return mean, X_uniques


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array given a target vector.

    Each category is encoded given the effect that it has on the
    target variable y. The method considers that categorical
    variables can present rare categories. It represents each category by the
    probability of y conditional on this category.
    In addition it takes an empirical Bayes approach to shrink the estimate.


    Parameters
    ----------
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the i-th
          column. The passed categories must be sorted and should not mix
          strings and numeric values.

        The categories used can be found in the ``categories_`` attribute.

    clf_type : string {'regression', 'binary-clf', 'multiclass-clf'}
        The type of classification/regression problem.

    dtype : number type, default np.float64
        Desired dtype of output.

    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.

    handle_missing : 'error' or '' (default)
        Whether to raise an error or impute with blank string '' if missing
        values (NaN) are present during fit (default is to impute).
        When this parameter is set to '', and a missing value is encountered
        during fit_transform, the resulting encoded columns for this feature
        will be all zeros.

    cross_val : bool, default=False
        Computing impact coded features on a subset of the data, using nested
        KFold cross-validation loop. Very useful when the number of
        observations of the encoded columns is high, to avoid overfitting.
        The number of inner and outer folds are defined by the ``n_folds`` and
        ``n_inner_folds`` parameters.

    n_folds : int, default=5
        The number of outer folds for the nested KFold. Useful only when
        ``cross_val`` is True.

    n_inner_folds: int, default=3
        The number of inner folds for the nested KFold. Useful only when
        ``cross_val`` is True.

    random_state: int, RandomState instance or None, default=None
        When ``cross_val`` is True, random_state affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise,
        this parameter has no effect. Pass an int for reproducible output
        across multiple function calls.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order corresponding with output of ``transform``).

    References
    -----------
    For more details, see Micci-Barreca, 2001: A preprocessing scheme for
    high-cardinality categorical attributes in classification and prediction
    problems.
    """
    def __init__(self,
                 categories='auto',
                 clf_type='binary-clf',
                 dtype=np.float64,
                 handle_unknown='error',
                 handle_missing='',
                 cross_val=False,
                 n_folds=5,
                 n_inner_folds=3,
                 random_state=1):
        self.categories = categories
        self.dtype = dtype
        self.clf_type = clf_type
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.cross_val = cross_val
        self.n_folds = n_folds
        self.n_inner_folds = n_inner_folds
        self.random_state = random_state

    def _more_tags(self):
        """
        Used internally by sklearn to ease the estimator checks.
        """
        return {"X_types": ["categorical"]}

    def fit(self, X, y):
        """Fit the TargetEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.
        y : array
            The associated target vector.

        Returns
        -------
        self
        """
        X = check_input(X)
        self.n_features_in_ = X.shape[1]
        if self.handle_missing not in ['error', '']:
            template = ("handle_missing should be either 'error' or "
                        "'', got %s")
            raise ValueError(template % self.handle_missing)
        if hasattr(X, 'iloc') and X.isna().values.any():
            if self.handle_missing == 'error':
                msg = ("Found missing values in input data; set "
                       "handle_missing='' to encode with missing values")
                raise ValueError(msg)
            if self.handle_missing != 'error':
                X = X.fillna(self.handle_missing)
        elif not hasattr(X, 'dtype') and isinstance(X, list):
            X = np.asarray(X, dtype=object)

        if hasattr(X, 'dtype'):
            mask = _object_dtype_isnan(X)
            if X.dtype.kind == 'O' and mask.any():
                if self.handle_missing == 'error':
                    msg = ("Found missing values in input data; set "
                           "handle_missing='' to encode with missing values")
                    raise ValueError(msg)
                if self.handle_missing != 'error':
                    X[mask] = self.handle_missing

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
        y_temp = check_array(y, dtype=None, ensure_2d=False)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
            y = check_array(y, dtype=np.object, ensure_2d=False)
        else:
            X = X_temp
            y = y_temp

        self.y = y

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
        self.n_ = len(y)
        if self.clf_type in ['binary-clf', 'regression']:
            self.Eyx_ = [{cat: np.mean(y[X[:, j] == cat])
                          for cat in self.categories_[j]}
                         for j in range(len(self.categories_))]
            self.Ey_ = np.mean(y)
            self.counter_ = {j: collections.Counter(X[:, j])
                             for j in range(n_features)}
        if self.clf_type in ['multiclass-clf']:
            self.classes_ = np.unique(y)

            self.Eyx_ = {c: [{cat: np.mean((y == c)[X[:, j] == cat])
                              for cat in self.categories_[j]}
                             for j in range(len(self.categories_))]
                         for c in self.classes_}
            self.Ey_ = {c: np.mean(y == c) for c in self.classes_}
            self.counter_ = {j: collections.Counter(X[:, j])
                             for j in range(n_features)}
        self.k_ = {j: len(self.counter_[j]) for j in self.counter_}
        return self

    def transform(self, X):
        """Transform X using specified encoding scheme.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features_new]
            The data to encode.

        Returns
        -------
        X_new : 2-d array
            Transformed input.
        """
        check_is_fitted(self, attributes=["n_features_in_"])
        X = check_input(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Number of features in the input data ({X.shape[1]}) does not match the number of features "
                f"seen during fit ({self.n_features_in_})."
            )
        if hasattr(X, 'iloc') and X.isna().values.any():
            if self.handle_missing == 'error':
                msg = ("Found missing values in input data; set "
                       "handle_missing='' to encode with missing values")
                raise ValueError(msg)
            if self.handle_missing != 'error':
                X = X.fillna(self.handle_missing)
        elif not hasattr(X, 'dtype') and isinstance(X, list):
            X = np.asarray(X, dtype=object)

        if hasattr(X, 'dtype'):
            mask = _object_dtype_isnan(X)
            if X.dtype.kind == 'O' and mask.any():
                if self.handle_missing == 'error':
                    msg = ("Found missing values in input data; set "
                           "handle_missing='' to encode with missing values")
                    raise ValueError(msg)
                if self.handle_missing != 'error':
                    X[mask] = self.handle_missing

        X_temp = check_array(X, dtype=None)
        y_temp = check_array(self.y, dtype=None, ensure_2d=False)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
            y = check_array(self.y, dtype=np.object, ensure_2d=False)
        else:
            X = X_temp
            y = y_temp

        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=int)
        X_mask = np.ones_like(X, dtype=bool)

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

        if self.cross_val is False:
            for j, cats in enumerate(self.categories_):
                unqX = np.unique(X[:, j])
                encoder = {x: 0 for x in unqX}
                if self.clf_type in ['binary-clf', 'regression']:
                    for x in unqX:
                        if x not in cats:
                            Eyx = 0
                        else:
                            Eyx = self.Eyx_[j][x]
                        lambda_n = lambda_(self.counter_[j][x], self.n_/self.k_[j])
                        encoder[x] = lambda_n*Eyx + (1 - lambda_n)*self.Ey_
                    x_out = np.zeros((len(X[:, j]), 1))
                    for i, x in enumerate(X[:, j]):
                        x_out[i, 0] = encoder[x]
                    out.append(x_out.reshape(-1, 1))
                if self.clf_type == 'multiclass-clf':
                    x_out = np.zeros((len(X[:, j]), len(self.classes_)))
                    lambda_n = {x: 0 for x in unqX}
                    for x in unqX:
                        lambda_n[x] = lambda_(self.counter_[j][x],
                                              self.n_/self.k_[j])
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

        if self.cross_val is True:
            np.random.seed(1)
            kf = KFold(n_splits=self.n_folds, shuffle=True,
                       random_state=self.random_state)
            # Global mean of the target, applied to unknown values
            global_mean = y.mean()
            split = 0
            out = np.zeros(shape=(n_samples, n_features))
            for j in range(n_features):
                impact_coded = pd.Series(dtype="float64")
                df_j = pd.DataFrame(X[:, j], columns=["features"])
                for infold, oof in kf.split(X[:, j]):
                    inner_means_df = pd.DataFrame()
                    infold_mean = y[infold].mean()
                    kf_inner = KFold(n_splits=self.n_inner_folds,
                                     shuffle=True, random_state=self.random_state)
                    inner_split = 0
                    for infold_inner, oof_inner in kf_inner.split(X[:, j][infold]):
                        X_a = X[:, j][infold][infold_inner]
                        y_a = y[infold][infold_inner]
                        infold_inner_mean, idx = arr_mean(X_a, y_a)
                        inner_means_df = inner_means_df.join(
                            pd.DataFrame(
                                infold_inner_mean, index=idx, columns=[inner_split]
                            ),
                            lsuffix=inner_split,
                            how="outer",
                        )
                        inner_means_df.fillna(infold_mean, inplace=True)
                        inner_split += 1
                    # Apply the mean of all infold_inner means
                    #  to the actual data, on oof
                    oof_X = df_j.iloc[oof]
                    inner_folds_mean = inner_means_df.mean(axis=1)
                    impact_coded_oof = (
                        oof_X["features"].map(inner_folds_mean).fillna(global_mean)
                    )
                    impact_coded = impact_coded.append(impact_coded_oof)
                    impact_coded.sort_index(inplace=True)
                    split += 1
                out[:, j] = impact_coded.to_numpy()
        return out
