import numpy as np
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
    check_is_fitted,
)

from . import _dataframe as sbd


class BlockNormalizerL2(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def fit(self, X):
        _ = self.fit_transform(X)
        return self

    def fit_transform(self, X):
        self._check_all_numeric(X)

        # Compute column-wise norm by filtering out nonfinite values.
        X = self._validate_data(X=X, accept_sparse=False, force_all_finite=False)

        self.avg_norm_ = _avg_norm(X)

        return X / self.avg_norm_

    def transform(self, X):
        check_is_fitted(self, "avg_norm_")
        X = self._validate_data(
            X=X, reset=False, accept_sparse=False, force_all_finite=False
        )
        return X / self.avg_norm_

    def _check_all_numeric(self, X):
        msg = "BlockNormalizer only accept numeric columns, but {col} is {dtype}."
        if hasattr(X, "__dataframe__"):
            for col in sbd.to_column_list(X):
                if not (sbd.is_numeric(col) or sbd.is_bool(col)):
                    raise ValueError(msg.format(col=col.name, dtype=sbd.dtype(col)))


def _avg_norm(X):
    # Sanity check: replace inf values with nan
    mask_finite = np.isfinite(X)
    X[~mask_finite] = np.nan

    # Only divide each column by the number finite values instead of the number
    # of samples.
    squared_diff = (X - np.nanmean(X, axis=0)) ** 2
    norm = np.nansum(squared_diff / mask_finite.sum(axis=0))

    # Avoid division by very small or zero values.
    if norm < 10 * np.finfo(norm.dtype).eps:
        norm = 1

    return np.sqrt(norm)
