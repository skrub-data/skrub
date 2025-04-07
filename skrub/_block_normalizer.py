import numpy as np
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
    check_is_fitted,
)

from . import _dataframe as sbd
from ._sklearn_compat import validate_data


class BlockNormalizerL2(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    r"""Fit the average L2 norm and apply the normalization.

    Compute the average L2 norm (a scalar) from a numerical DataFrame or a 2D NumPy
    array, and use this norm to normalize the input element-wise. This computation
    is robust to non-finite values such as ``np.inf`` or ``np.nan``, but raises an
    error if string values are present.

    Attributes
    ----------
    avg_norm_ : float
        The average l2 norm, computed on the training set.

    See Also
    --------
    :class:`~sklearn.preprocessing.Normalizer` :
        Performs row-wise normalization.

    Notes
    -----
    We define this norm as:

    .. math::

        ||X||_B = \sqrt{\sum_{j=1}^D \sigma^2(X_j)} \in \mathbb{R}

    where:

    * :math:`\sigma^2(X_j)` is the population variance of the column :math:`j`
    * :math:`D` is the number of features

    When :math:`D = 1`, this norm is the population standard deviation of the column.
    We then normalize each element as:

    .. math::

        \forall i,j, \quad \tilde{X}_{ij} = \frac{X_{ij}}{||X||_B}
    """

    def fit(self, X, y=None):
        """Fit the normalizer.

        Parameters
        ----------
        X : array-like, of shape (n_samples, n_features)
            The data used to compute the norm.
        y : None
            Unused. Here for compatibility with scikit-learn.

        Returns
        -------
        BlockNormalizerL2
            The fitted BlockNormalizerL2 instance (self).
        """
        self._check_all_numeric(X)

        # Compute column-wise norm by filtering out nonfinite values.
        X = validate_data(self, X=X, accept_sparse=False, ensure_all_finite=False)

        self.avg_norm_ = _avg_norm(X)

        return self

    def transform(self, X):
        """Normalize the data.

        Parameters
        ----------
        X : array-like, of shape (n_samples, n_features)
            The data to normalize.

        Returns
        -------
        X_out : numpy array of shape (n_samples, n_features)
            The normalized data.
        """
        check_is_fitted(self, "avg_norm_")
        X = validate_data(
            self, X=X, reset=False, accept_sparse=False, ensure_all_finite=False
        )
        return X / self.avg_norm_

    def _check_all_numeric(self, X):
        if hasattr(X, "__dataframe__"):
            msg = "BlockNormalizer only accept numeric columns, but {col} is {dtype}."
            for col in sbd.to_column_list(X):
                if not (sbd.is_numeric(col) or sbd.is_bool(col)):
                    raise ValueError(msg.format(col=col.name, dtype=sbd.dtype(col)))
        elif isinstance(X, np.ndarray):
            msg = (
                "BlockNormalizer only accept numeric values, but the array has "
                "at least one non numeric value."
            )
            try:
                X.astype("float32")
            except ValueError as e:
                raise ValueError(msg) from e


def _avg_norm(X):
    # Sanity check: replace inf values with nan
    mask_finite = np.isfinite(X)
    X[~mask_finite] = np.nan

    norm = np.sqrt(np.nanvar(X, ddof=0, axis=0).sum())

    # Avoid division by very small or zero values.
    if norm < 10 * np.finfo(norm.dtype).eps:
        norm = 1

    return norm
