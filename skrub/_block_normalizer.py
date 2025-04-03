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

        self.avg_norm_ = avg_norm(X)

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


def avg_norm(X):
    n_features = X.shape[1]
    norm = 0
    mask_finite = np.isfinite(X)
    for jdx in range(n_features):
        m = mask_finite[:, jdx]
        x = X[:, jdx][m]
        norm += ((x - x.mean()) ** 2).sum(axis=None)
    n_sample_finite = mask_finite.sum() / n_features
    norm /= n_sample_finite

    # Avoid division by very small or zero values.
    if norm < 10 * np.finfo(norm.dtype).eps:
        norm = 1

    return np.sqrt(norm)
