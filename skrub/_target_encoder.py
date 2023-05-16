import collections
from typing import Dict, List, Literal, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from sklearn.utils.fixes import _object_dtype_isnan
from sklearn.utils.validation import check_is_fitted

from skrub._utils import check_input


def lambda_(x, n):
    return x / (x + n)


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array given a target vector.

    Each category is encoded given the effect that it has in the
    target variable :term:`y`. The method considers that categorical
    variables can present rare categories. It represents each category by the
    probability of :term:`y` conditional on this category.
    In addition, it takes an empirical Bayes approach to shrink the estimate.

    Parameters
    ----------
    categories : 'auto' or list of list of int or str
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : `categories[i]` holds the categories expected in the `i`-th
          column. The passed categories must be sorted and should not mix
          strings and numeric values.

        The categories used can be found in the ``categories_`` attribute.
    clf_type : {'regression', 'binary-clf', 'multiclass-clf'}, default='binary-clf'
        The type of classification/regression problem.
    dtype : number type, default=np.float64
        Desired dtype of output.
    handle_unknown : {'error', 'ignore'}, default='error'
        Whether to raise an error or ignore if an unknown categorical feature is
        present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the encoded columns for this feature
        will be assigned the prior mean of the target variable.
    handle_missing : {'error', ''}, default=''
        Whether to raise an error or impute with blank string '' if missing
        values (NaN) are present during :func:`~TargetEncoder.fit`
        (default is to impute).
        When this parameter is set to '', and a missing value is encountered
        during :func:`~TargetEncoder.fit_transform`, the resulting encoded
        columns for this feature will be all zeros.

    Attributes
    ----------
    n_features_in_ : int
        Number of features in the data seen during :func:`~TargetEncoder.fit`.
    categories_ : list of :obj:`~numpy.ndarray`
        The categories of each feature determined during :func:`~TargetEncoder.fit`
        (in order corresponding with output of :func:`~TargetEncoder.transform`).
    n_ : int
        Length of :term:`y`

    See Also
    --------
    :class:`skrub.GapEncoder`
        Encodes dirty categories (strings) by constructing latent topics with
        continuous encoding.
    :class:`skrub.MinHashEncoder`
        Encode string columns as a numeric array with the minhash method.
    :class:`skrub.SimilarityEncoder`
        Encode string columns as a numeric array with n-gram string similarity.

    References
    ----------
    For more details, see Micci-Barreca, 2001: A preprocessing scheme for
    high-cardinality categorical attributes in classification and prediction
    problems.

    Examples
    --------
    >>> enc = TargetEncoder(handle_unknown='ignore')
    >>> X = [['male'], ['Male'], ['Female'], ['male'], ['Female']]
    >>> y = np.array([1, 2, 3, 4, 5])

    >>> enc.fit(X, y)
    TargetEncoder(handle_unknown='ignore')

    The encoder has found the following categories:

    >>> enc.categories_
    [array(['Female', 'Male', 'male'], dtype='<U6')]

    We will encode the following categories, of which the first two are unknown :

    >>> X2 = [['MALE'], ['FEMALE'], ['Female'], ['male'], ['Female']]

    >>> enc.transform(X2)
    array([[3.        ],
        [3.        ],
        [3.54545455],
        [2.72727273],
        [3.54545455]])

    As expected, they were encoded according to their influence on y.
    The unknown categories were assigned the mean of the target variable.
    """

    n_features_in_: int
    _label_encoders_: List[LabelEncoder]
    categories_: List[np.ndarray]
    n_: int

    def __init__(
        self,
        categories: Union[Literal["auto"], List[Union[List[str], np.ndarray]]] = "auto",
        clf_type: Literal["regression", "binary-clf", "multiclass-clf"] = "binary-clf",
        dtype: type = np.float64,
        handle_unknown: Literal["error", "ignore"] = "error",
        handle_missing: Literal["error", ""] = "",
    ):
        self.categories = categories
        self.dtype = dtype
        self.clf_type = clf_type
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing

    def _more_tags(self) -> Dict[str, List[str]]:
        """
        Used internally by sklearn to ease the estimator checks.
        """
        return {"X_types": ["categorical"]}

    def fit(self, X, y) -> "TargetEncoder":
        """Fit the instance to `X`.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.
        y : :obj:`~numpy.ndarray`
            The associated target vector.

        Returns
        -------
        :obj:`~skrub.TargetEncoder`
            Fitted :class:`~skrub.TargetEncoder` instance (self).
        """
        X = check_input(X)
        self.n_features_in_ = X.shape[1]
        if self.handle_missing not in ["error", ""]:
            raise ValueError(
                f"Got handle_missing={self.handle_missing!r}, but expected "
                "any of {'error', ''}. "
            )

        mask = _object_dtype_isnan(X)
        if mask.any():
            if self.handle_missing == "error":
                raise ValueError(
                    "Found missing values in input data; set "
                    "handle_missing='' to encode with missing values. "
                )
            else:
                X[mask] = self.handle_missing

        if self.handle_unknown not in ["error", "ignore"]:
            raise ValueError(
                f"Got handle_unknown={self.handle_unknown!r}, but expected "
                "any of {'error', 'ignore'}. "
            )

        if self.categories != "auto":
            for cats in self.categories:
                if not np.all(np.sort(cats) == np.array(cats)):
                    raise ValueError("Unsorted categories are not yet supported. ")

        X_temp = check_array(X, dtype=None)
        X = X_temp

        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for j in range(n_features):
            le = self._label_encoders_[j]
            Xj = X[:, j]
            if self.categories == "auto":
                le.fit(Xj)
            else:
                if self.handle_unknown == "error":
                    valid_mask = np.in1d(Xj, self.categories[j])
                    if not np.all(valid_mask):
                        diff = np.unique(Xj[~valid_mask])
                        raise ValueError(
                            f"Found unknown categories {diff} in column {j} during fit"
                        )
                le.classes_ = np.array(self.categories[j])

        self.categories_ = [le.classes_ for le in self._label_encoders_]
        self.n_ = len(y)
        if self.clf_type in ["binary-clf", "regression"]:
            self.Eyx_ = [
                {cat: np.mean(y[X[:, j] == cat]) for cat in self.categories_[j]}
                for j in range(len(self.categories_))
            ]
            self.Ey_ = np.mean(y)
            self.counter_ = {j: collections.Counter(X[:, j]) for j in range(n_features)}
        if self.clf_type in ["multiclass-clf"]:
            self.classes_ = np.unique(y)

            self.Eyx_ = {
                c: [
                    {
                        cat: np.mean((y == c)[X[:, j] == cat])
                        for cat in self.categories_[j]
                    }
                    for j in range(len(self.categories_))
                ]
                for c in self.classes_
            }
            self.Ey_ = {c: np.mean(y == c) for c in self.classes_}
            self.counter_ = {j: collections.Counter(X[:, j]) for j in range(n_features)}
        self.k_ = {j: len(self.counter_[j]) for j in self.counter_}
        return self

    def transform(self, X) -> np.ndarray:
        """Transform `X` using the specified encoding scheme.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features_new]
            The data to encode.

        Returns
        -------
        2-d :class:`~numpy.ndarray`
            Transformed input.
        """
        check_is_fitted(self, attributes=["n_features_in_"])
        X = check_input(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"The number of features in the input data ({X.shape[1]}) "
                "does not match the number of features "
                f"seen during fit ({self.n_features_in_})."
            )
        mask = _object_dtype_isnan(X)
        if mask.any():
            if self.handle_missing == "error":
                raise ValueError(
                    "Found missing values in input data; set "
                    "handle_missing='' to encode with missing values. "
                )
            else:
                X[mask] = self.handle_missing

        X_temp = check_array(X, dtype=None)
        X = X_temp

        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=int)
        X_mask = np.ones_like(X, dtype=bool)

        for i in range(n_features):
            Xi = X[:, i]
            valid_mask = np.in1d(Xi, self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == "error":
                    diff = np.unique(X[~valid_mask, i])
                    raise ValueError(
                        f"Found unknown categories {diff} in column {i} "
                        "during transform."
                    )
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
            encoder = {x: 0 for x in unqX}
            if self.clf_type in ["binary-clf", "regression"]:
                for x in unqX:
                    if x not in cats:
                        Eyx = 0
                    else:
                        Eyx = self.Eyx_[j][x]
                    lambda_n = lambda_(self.counter_[j][x], self.n_ / self.k_[j])
                    encoder[x] = lambda_n * Eyx + (1 - lambda_n) * self.Ey_
                x_out = np.zeros((len(X[:, j]), 1))
                for i, x in enumerate(X[:, j]):
                    x_out[i, 0] = encoder[x]
                out.append(x_out.reshape(-1, 1))
            if self.clf_type == "multiclass-clf":
                x_out = np.zeros((len(X[:, j]), len(self.classes_)))
                lambda_n = {x: 0 for x in unqX}
                for x in unqX:
                    lambda_n[x] = lambda_(self.counter_[j][x], self.n_ / self.k_[j])
                for k, c in enumerate(np.unique(self.classes_)):
                    for x in unqX:
                        if x not in cats:
                            Eyx = 0
                        else:
                            Eyx = self.Eyx_[c][j][x]
                        encoder[x] = lambda_n[x] * Eyx + (1 - lambda_n[x]) * self.Ey_[c]
                    for i, x in enumerate(X[:, j]):
                        x_out[i, k] = encoder[x]
                out.append(x_out)
        out = np.hstack(out)
        return out
