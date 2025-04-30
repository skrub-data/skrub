import numbers

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._on_each_column import OnEachColumn, RejectColumn, SingleColumnTransformer


class SingleColumnSquashingScaler(SingleColumnTransformer):
    r"""Preprocess a numerical column by robust centering, scaling, and soft clipping to a defined interval.

    This transformer has two stages:

    1. Center the median of the data to zero
       and multiply the data by a scaling factor
       determined based on quantiles of the distribution.
       This is similar to scikit-learn's RobustScaler
       but with a min-max-scaling based handling
       for edge cases in which the two quantiles are equal.
    2. Apply a soft-clipping function to limit the data
       to the interval [-max_absolute_value, max_absolute_value]
       in an injective way.

    Infinite values will be mapped to the corresponding boundaries of the interval.
    NaN values will be preserved.

    Parameters
    ----------
    max_absolute_value : float, default=3.0
        Maximum absolute value that the transformed data can take.
    lower_quantile : float, default=0.25
        Value of the lower quantile that will be used
        to determine the scaling factor to rescale the data.
    upper_quantile : float, default=0.75
        Value of the upper quantile that will be used
        to determine the scaling factor to rescale the data.

    Notes
    -----

    The formula for the transform is

    .. math::

        \begin{align*}
            a &:= \begin{cases}
                1/(q_{\beta} - q_{\alpha}) &, q_{\beta} \neq q_{\alpha} \\
                2/(q_1 - q_0) &, q_{\beta} = q_{\alpha} \text{ and } q_1 \neq q_0 \\
                0 &, \text{ otherwise.}
            \end{cases} \\
            z &:= a(x - q_{1/2}), \\
            x_{\mathrm{out}} &:= \frac{z}{\sqrt{1 + (z/B)^2}},
        \end{align*}

    where

    - :math:`x` is a value in the input column.
    - :math:`q_{\gamma}` is the :math:`\gamma`-quantile of the finite values in X,
    - :math:`B` is max_abs_value
    - :math:`\alpha` is lower_quantile
    - :math:`\beta` is upper_quantile.
    - :math:`x_{\mathrm{out}}` is the result of the computation that will be returned.

    References
    ----------
    This method has been introduced as the robust scaling and smooth clipping transform in
    `Better by default: Strong pre-tuned MLPs and boosted trees on tabular data
    <https://arxiv.org/abs/2407.04491>`_ by Holzmüller, Grinsztajn, Steinwart (2024).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from skrub import SingleColumnSquashingScaler

    >>> tfm = SingleColumnSquashingScaler()
    >>> X = pd.Series([-np.inf, -100.0, 0.0, 8.0, 9.0, 10.0, 20.0, 100.0, np.inf, np.nan], name='improvement')
    >>> tfm.fit_transform(X)
    0   -3.000000
    1   -2.871295
    2   -0.789352
    3   -0.090867
    4    0.000000
    5    0.090867
    6    0.948683
    7    2.820284
    8    3.000000
    9         NaN
    Name: improvement, dtype: float32
    """  # noqa: E501

    def __init__(
        self,
        max_absolute_value=3.0,
        lower_quantile=0.25,
        upper_quantile=0.75,
    ):
        self.max_absolute_value = max_absolute_value
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

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

        if not isinstance(self.max_absolute_value, numbers.Number):
            raise TypeError(
                "max_absolute_value needs to be a number, but got"
                f" type(max_absolute_value)={type(self.max_absolute_value)}"
            )
        if not isinstance(self.lower_quantile, numbers.Number):
            raise TypeError(
                "lower_quantile needs to be a number, but got"
                f" type(lower_quantile)={type(self.lower_quantile)}"
            )
        if not isinstance(self.upper_quantile, numbers.Number):
            raise TypeError(
                "upper_quantile needs to be a number, but got"
                f" type(upper_quantile)={type(self.upper_quantile)}"
            )
        if self.max_absolute_value <= 0 or not np.isfinite(self.max_absolute_value):
            raise ValueError(
                "max_absolute_value needs to be a positive finite float, but got"
                f" {self.max_absolute_value}"
            )
        if not (0 <= self.lower_quantile < self.upper_quantile <= 1):
            raise ValueError(
                "need 0 <= lower_quantile < upper_quantile <= 1, but got"
                f" lower_quantile={self.lower_quantile} and"
                f" upper_quantile={self.upper_quantile}"
            )

        self.col_name_ = sbd.name(X)

        if not sbd.is_numeric(X):
            raise RejectColumn(f"Column {self.col_name_} is not numeric.")

        eps = np.finfo("float32").tiny

        values = sbd.to_numpy(X).astype(np.float32)
        finite_values = values[np.isfinite(values)]
        if len(finite_values) > 0:
            self.median_ = np.median(finite_values)
            quantiles = np.quantile(
                finite_values, [self.lower_quantile, self.upper_quantile]
            )
            if quantiles[1] == quantiles[0]:
                min = np.min(finite_values)
                max = np.max(finite_values)
                if min == max:
                    self.scale_ = 0
                else:
                    self.scale_ = 2.0 / (max - min + eps)
            else:
                self.scale_ = 1.0 / (quantiles[1] - quantiles[0] + eps)
        else:
            self.median_ = 0.0
            self.scale_ = 1.0

        return self.transform(X)

    def transform(self, X):
        """Transform a column.

        Parameters
        ----------
        X : Pandas or Polars series
            The column to transform.

        Returns
        -------
        result: Pandas or Polars column with shape (len(X),)
            The scaled and squashed representation of the input.
        """

        values_np = sbd.to_numpy(X).astype(np.float32, copy=True)
        isfinite = np.isfinite(values_np)
        scaled_finite = self.scale_ * (values_np[isfinite] - self.median_)
        values_np[isfinite] = scaled_finite / np.sqrt(
            1 + (scaled_finite / self.max_absolute_value) ** 2
        )
        isinf = np.isinf(values_np)
        values_np[isinf] = np.sign(values_np[isinf]) * self.max_absolute_value

        # this will use the column name of the column that was passed in fit_transform,
        # not the name of X
        result = sbd.make_column_like(X, values_np, self.col_name_)
        result = sbd.copy_index(X, result)

        return result


class SquashingScaler(BaseEstimator, TransformerMixin):
    r"""Preprocess numerical columns by robust centering, scaling, and soft clipping to a defined interval.

        This transformer is applied to each column independently.
        It uses two stages:

        1. Center the median of the data to zero
           and multiply the data by a scaling factor
           determined based on quantiles of the distribution.
           This is similar to scikit-learn's RobustScaler
           but with a min-max-scaling based handling
           for edge cases in which the two quantiles are equal.
        2. Apply a soft-clipping function to limit the data
           to the interval [-max_absolute_value, max_absolute_value]
           in an injective way.

        Infinite values will be mapped to the corresponding boundaries of the interval.
        NaN values will be preserved.

        Parameters
        ----------
        max_absolute_value : float, default=3.0
            Maximum absolute value that the transformed data can take.
        lower_quantile : float, default=0.25
            Value of the lower quantile that will be used
            to determine the scaling factor to rescale the data.
        upper_quantile : float, default=0.75
            Value of the upper quantile that will be used
            to determine the scaling factor to rescale the data.

        Notes
        -----

        The formula for the transform is

        .. math::

            \begin{align*}
                a &:= \begin{cases}
                    1/(q_{\beta} - q_{\alpha}) &, q_{\beta} \neq q_{\alpha} \\
                    2/(q_1 - q_0) &, q_{\beta} = q_{\alpha} \text{ and } q_1 \neq q_0 \\
                    0 &, \text{ otherwise.}
                \end{cases} \\
                z &:= a(x - q_{1/2}), \\
                x_{\mathrm{out}} &:= \frac{z}{\sqrt{1 + (z/B)^2}},
            \end{align*}

        where

        - :math:`x` is a value in the input column.
        - :math:`q_{\gamma}` is the :math:`\gamma`-quantile of the finite values in X,
        - :math:`B` is max_abs_value
        - :math:`\alpha` is lower_quantile
        - :math:`\beta` is upper_quantile.
        - :math:`x_{\mathrm{out}}` is the result of the computation that will be returned.

        References
        ----------
        This method has been introduced as the robust scaling and smooth clipping transform in
        `Better by default: Strong pre-tuned MLPs and boosted trees on tabular data
        <https://arxiv.org/abs/2407.04491>`_ by Holzmüller, Grinsztajn, Steinwart (2024).

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from skrub import SingleColumnSquashingScaler

        >>> tfm = SquashingScaler()
        >>> X = pd.Series([-np.inf, -100.0, 0.0, 8.0, 9.0, 10.0, 20.0, 100.0, np.inf, np.nan], name='improvement')
        >>> tfm.fit_transform(X)
        0   -3.000000
        1   -2.871295
        2   -0.789352
        3   -0.090867
        4    0.000000
        5    0.090867
        6    0.948683
        7    2.820284
        8    3.000000
        9         NaN
        Name: improvement, dtype: float32
        """  # noqa: E501

    def __init__(
        self,
        max_absolute_value=3.0,
        lower_quantile=0.25,
        upper_quantile=0.75,
    ):
        self.max_absolute_value = max_absolute_value
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        """Fit the transformer and transform a column.

        Parameters
        ----------
        X : Pandas or Polars dataframe or series
            The column to transform.
        y : None
            Unused. Here for compatibility with scikit-learn.

        Returns
        -------
        self: The current object.
        """
        self.tfm_ = SingleColumnSquashingScaler(
            max_absolute_value=self.max_absolute_value,
            lower_quantile=self.lower_quantile,
            upper_quantile=self.upper_quantile,
        )
        if not sbd.is_column(X):
            self.tfm_ = OnEachColumn(self.tfm_)
        self.tfm_.fit(X)
        return self

    def transform(self, X):
        """Transform a dataframe.

        Parameters
        ----------
        X : Pandas or Polars dataframe or series
            The data to transform.

        Returns
        -------
        result: Pandas or Polars dataframe or series with the same number of rows as X.
            The transformed version of the input.
        """
        check_is_fitted(self)
        return self.tfm_.transform(X)
