import numbers

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.utils._array_api import get_namespace
from sklearn.utils.validation import check_is_fitted


def _soft_clip(X, max_absolute_value=3.0):
    """Apply a soft clipping to the data.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to be clipped.
    max_absolute_value : float, default=3.0
        Maximum absolute value that the transformed data can take.

    Returns
    -------
    X_clipped : array-like, shape (n_samples, n_features)
        The clipped version of the input.
    """
    xp, _ = get_namespace(X)
    return X / xp.sqrt(1 + (X / max_absolute_value) ** 2)


class SquashingScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    r"""Perform robust centering and scaling of followed by soft clipping.

    When features have large outliers, smooth clipping prevents the outliers from
    affecting the result too strongly, while robust scaling prevents the outliers from
    affecting the inlier scaling. Infinite values are mapped to the corresponding boundaries of the interval. NaN
    values are preserved.

    Parameters
    ----------
    max_absolute_value : float, default=3.0
        Maximum absolute value that the transformed data can take.

    quantile_range : tuple of float, default=(0.25, 0.75)
        The quantiles used to compute the scaling factor. The first value is the lower
        quantile and the second value is the upper quantile. The default values are
        the 25th and 75th percentiles, respectively. The quantiles are used
        to compute the scaling factor for the robust scaling step. The quantiles are
        computed from the finite values in the input column. If the two quantiles are
        equal, the scaling factor is computed from the 0th and 100th percentiles
        (i.e., the minimum and maximum values of the finite values in the input column).

    copy : bool, default=True
        Whether to copy the input data or not. If set to False, the input data will
        be modified in-place. This is useful for memory efficiency, but may lead to
        unexpected results if the input data is used later in the code. If set to True,
        a copy of the input data will be made, and the original data will remain
        unchanged. This is the default behavior and is recommended for most use cases.

    Notes
    -----
    This transformer is applied to each column independently. It uses two stages:

    1. The first stage centers the median of the data to zero and multiply the data by a
       scaling factor determined from quantiles of the distribution, using
        scikit-learn's :`~sklearn.preprocessing.RobustScaler`. It also handles
        edge-cases in which the two quantiles are equal by following-up with a
        :class:`~sklearn.preprocessing.MinMaxEncoder`.
    2. The second stage applies a soft clipping to the transformed data to limit the
       data to the interval ``[-max_absolute_value, max_absolute_value]`` in an
       injective way.

    Infinite values will be mapped to the corresponding boundaries of the interval. NaN
    values will be preserved.

    The formula for the transform is:

    .. math::

        \begin{align*}
            a &:= \begin{cases}
                1/(q_{\beta} - q_{\alpha}) &\mathrm{if} q_{\beta} \neq q_{\alpha} \\
                2/(q_1 - q_0) &\mathrm{if} q_{\beta} = q_{\alpha} \text{ and } q_1 \neq
                q_0 \\ 0 &\mathrm{if} \text{ otherwise.}
            \end{cases} \\ z &:= a(x - q_{1/2}), \\ x_{\mathrm{out}} &:=
            \frac{z}{\sqrt{1 + (z/B)^2}},
        \end{align*}

    where:

    - :math:`x` is a value in the input column.
    - :math:`q_{\gamma}` is the :math:`\gamma`-quantile of the finite values in X,
    - :math:`B` is max_abs_value
    - :math:`\alpha` is lower_quantile
    - :math:`\beta` is upper_quantile.

    References
    ----------
    This method has been introduced as the robust scaling and smooth clipping transform
    in `Better by default: Strong pre-tuned MLPs and boosted trees on tabular data
    <https://arxiv.org/abs/2407.04491>`_ by Holzmüller, Grinsztajn, Steinwart (2024).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from skrub import SquashingScaler
    >>> X = pd.DataFrame(dict(a=[-100.0, 0.0, 8.0, 9.0, 10.0, 20.0, 100.0, np.nan]))
    >>> SquashingScaler().fit_transform(X)
    array([[-2.98982558],
          [-2.12132034],
          [-0.33129458],
          [ 0.        ],
          [ 0.33129458],
          [ 2.3218719 ],
          [ 2.98543462],
          [        nan]])
    """  # noqa: E501

    def __init__(
        self,
        max_absolute_value=3.0,
        quantile_range=(0.25, 0.75),
        copy=True,
    ):
        self.max_absolute_value = max_absolute_value
        self.quantile_range = quantile_range
        self.copy = copy

    def fit_transform(self, X, y=None):
        """Fit the transformer and transform a column.

        Parameters
        ----------
        X : Pandas or Polars series
            The column to transform.
        y : None
            Unused. Here for compatibility with scikit-learn.

        Returns
        -------
        X_out: Pandas or Polars series with shape (len(X),)
            The transformed version of the input.
        """
        del y

        if not (
            isinstance(self.max_absolute_value, numbers.Number)
            and self.max_absolute_value > 0
            and np.isfinite(self.max_absolute_value)
        ):
            raise TypeError(
                f"Got {self.max_absolute_value=!r}, but expected a positive finite "
                "float."
            )

        self.robust_scaler_ = RobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=self.quantile_range,
            copy=self.copy,
        )
        X_tr = self.robust_scaler_.fit_transform(X)

        if (minmax_indices := np.argwhere(self.robust_scaler_.scale_ == 1)).any():
            # if the scale is 1, we can use a min-max scaler to handle the edge cases
            self.minmax_scaler_ = MinMaxScaler(
                feature_range=(-1, 1),
                copy=self.copy,
                clip=False,
            )
            X_tr[:, minmax_indices] = self.minmax_scaler_.fit_transform(
                X_tr[:, minmax_indices]
            )
            self.minmax_indices_ = minmax_indices
        else:
            self.minmax_scaler_ = None
            self.minmax_indices_ = None

        print(X)
        return _soft_clip(X_tr, self.max_absolute_value)

    def transform(self, X):
        """Transform a column.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_out: array-like, shape (n_samples, n_features)
            The transformed version of the input.
        """
        check_is_fitted(self, ["robust_scaler_", "minmax_scaler_"])

        X_tr = self.robust_scaler_.transform(X)

        if self.minmax_scaler_ is not None:
            X_tr[:, self.minmax_indices_] = self.minmax_scaler_.transform(
                X_tr[:, self.minmax_indices_]
            )

        return _soft_clip(X_tr, self.max_absolute_value)
