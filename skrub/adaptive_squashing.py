import numbers

import numpy as np

from . import _dataframe as sbd
from ._on_each_column import RejectColumn, SingleColumnTransformer


class AdaptiveSquashingTransformer(SingleColumnTransformer):
    def __init__(
        self,
        squash_threshold=3.0,
        lower_quantile=0.25,
        upper_quantile=0.75,
    ):
        self.squash_threshold = squash_threshold
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

        if not isinstance(self.squash_threshold, numbers.Number):
            raise ValueError(
                "squash_threshold needs to be a number, but got"
                f" type(squash_threshold)={type(self.squash_threshold)}"
            )
        if not isinstance(self.lower_quantile, numbers.Number):
            raise ValueError(
                "lower_quantile needs to be a number, but got"
                f" type(lower_quantile)={type(self.lower_quantile)}"
            )
        if not isinstance(self.upper_quantile, numbers.Number):
            raise ValueError(
                "upper_quantile needs to be a number, but got"
                f" type(upper_quantile)={type(self.upper_quantile)}"
            )
        if self.squash_threshold <= 0 or not np.isfinite(self.squash_threshold):
            raise ValueError(
                "squash_threshold needs to be a positive finite float, but got"
                f" {self.squash_threshold}"
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

        if not self.col_name_:
            self.col_name_ = "adaptive_squashed"

        eps = 1e-30  # todo: expose?

        values = sbd.to_numpy(X).astype(np.float32)
        finite_values = values[np.isfinite(values)]
        if len(finite_values) > 0:
            self.median_ = np.median(finite_values)
            quantiles = [
                np.quantile(finite_values, q)
                for q in [self.lower_quantile, self.upper_quantile]
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
            1 + (scaled_finite / self.squash_threshold) ** 2
        )
        isinf = np.isinf(values_np)
        result[isinf] = np.sign(values_np[isinf]) * self.squash_threshold

        # this will use the column name of the column that was passed in fit_transform,
        # not the name of X
        return self._post_process(X, result, self.col_name_)

    def _post_process(self, X, result, name):
        result = sbd.make_column_like(X, result, name)
        result = sbd.copy_index(X, result)

        return result
