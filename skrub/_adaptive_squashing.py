import numbers

import numpy as np

from . import _dataframe as sbd
from ._on_each_column import RejectColumn, SingleColumnTransformer


class AdaptiveSquashingTransformer(SingleColumnTransformer):
    """Preprocess a numerical column by robust centering, scaling, \
    and soft clipping to a defined interval.

    This class implements the robust scaling and smooth clipping transformation
    proposed for tabular neural networks by
    David Holzmüller, Léo Grinsztajn, and Ingo Steinwart,
    Better by default: Strong pre-tuned MLPs and boosted trees on tabular data,
    Advances in Neural Information Processing Systems, 2024.

    It will
    1) center the median of the data to zero and multiply the data by a scaling factor
     determined based on quantiles of the distribution.
     This is similar to scikit-learn's QuantileTransformer
     but with a min-max-scaling based handling
     for edge cases in which the two quantiles are equal.
    2) apply a soft-clipping function to limit the data
     to the interval [-max_absolute_value, max_absolute_value]
     in an injective way.

    Infinite values will be mapped to the corresponding boundaries of the interval.
    NaN values will be preserved.

    The output feature will be named ``{col_name}`` if the series has a name,
    and ``adaptive_squashed`` if it does not.

    Parameters
    ----------
    max_absolute_value : float, default=3.0
        Maximum absolute value that the transformed data can take.
    lower_quantile_alpha : float, default=0.25
        Value of the lower quantile that will be used
        to determine the scaling factor to rescale the data.
    upper_quantile_alpha : float, default=0.75
        Value of the upper quantile that will be used
        to determine the scaling factor to rescale the data.

    References
    ----------
    For a detailed description of the method, see
    `Better by default: Strong pre-tuned MLPs and boosted trees on tabular data
    <https://arxiv.org/abs/2407.04491>`_ by Holzmüller, Grinsztajn, Steinwart (2024).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from skrub import AdaptiveSquashingTransformer

    >>> tfm = AdaptiveSquashingTransformer()
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
        lower_quantile_alpha=0.25,
        upper_quantile_alpha=0.75,
    ):
        self.max_absolute_value = max_absolute_value
        self.lower_quantile_alpha = lower_quantile_alpha
        self.upper_quantile_alpha = upper_quantile_alpha

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
            raise ValueError(
                "max_absolute_value needs to be a number, but got"
                f" type(max_absolute_value)={type(self.max_absolute_value)}"
            )
        if not isinstance(self.lower_quantile_alpha, numbers.Number):
            raise ValueError(
                "lower_quantile_alpha needs to be a number, but got"
                f" type(lower_quantile_alpha)={type(self.lower_quantile_alpha)}"
            )
        if not isinstance(self.upper_quantile_alpha, numbers.Number):
            raise ValueError(
                "upper_quantile_alpha needs to be a number, but got"
                f" type(upper_quantile_alpha)={type(self.upper_quantile_alpha)}"
            )
        if self.max_absolute_value <= 0 or not np.isfinite(self.max_absolute_value):
            raise ValueError(
                "max_absolute_value needs to be a positive finite float, but got"
                f" {self.max_absolute_value}"
            )
        if not (0 <= self.lower_quantile_alpha < self.upper_quantile_alpha <= 1):
            raise ValueError(
                "need 0 <= lower_quantile_alpha < upper_quantile_alpha <= 1, but got"
                f" lower_quantile_alpha={self.lower_quantile_alpha} and"
                f" upper_quantile_alpha={self.upper_quantile_alpha}"
            )

        self.col_name_ = sbd.name(X)

        if not sbd.is_numeric(X):
            raise RejectColumn(f"Column {self.col_name_} is not numeric.")

        if not self.col_name_:
            self.col_name_ = "adaptive_squashed"

        eps = 1e-30  # todo: expose?

        values = sbd.to_numpy(X).astype(np.float32)
        finite_values = values[np.isfinite(values)]
        if len(finite_values) > 0:
            self.median_ = np.median(finite_values)
            quantiles = [
                np.quantile(finite_values, q)
                for q in [self.lower_quantile_alpha, self.upper_quantile_alpha]
            ]
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

        self._is_fitted = True
        self.n_components_ = 1

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

        values_np = sbd.to_numpy(X).astype(np.float32)
        result = np.copy(values_np)
        isfinite = np.isfinite(values_np)
        scaled_finite = self.scale_ * (values_np[isfinite] - self.median_)
        result[isfinite] = scaled_finite / np.sqrt(
            1 + (scaled_finite / self.max_absolute_value) ** 2
        )
        isinf = np.isinf(values_np)
        result[isinf] = np.sign(values_np[isinf]) * self.max_absolute_value

        # this will use the column name of the column that was passed in fit_transform,
        # not the name of X
        return self._post_process(X, result, self.col_name_)

    def _post_process(self, X, result, name):
        result = sbd.make_column_like(X, result, name)
        result = sbd.copy_index(X, result)

        return result


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    from skrub import AdaptiveSquashingTransformer

    tfm = AdaptiveSquashingTransformer()
    X = pd.Series(
        [-np.inf, -100.0, 0.0, 8.0, 9.0, 10.0, 20.0, 100.0, np.inf, np.nan],
        name="improvement",
    )
    tfm.fit_transform(X)
