import pandas as pd
from sklearn.base import BaseEstimator


class ToNumericCol(BaseEstimator):
    def fit_transform(self, column):
        try:
            transformed = pd.to_numeric(column)
            self.output_dtype_ = transformed.dtype
            return transformed
        except Exception:
            raise NotImplementedError()

    def transform(self, column):
        return pd.to_numeric(column).astype(self.output_dtype_)
