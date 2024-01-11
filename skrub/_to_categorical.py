import pandas as pd
from pandas.api.types import CategoricalDtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator

_HIGH_CARD_THRESHOLD = 30


class ToCategoricalCol(BaseEstimator):
    def fit_transform(self, column):
        if isinstance(column.dtype, CategoricalDtype):
            self.output_dtype_ = column.dtype
            return column
        if is_numeric_dtype(column) or is_datetime64_any_dtype(column):
            raise NotImplementedError()
        categories = list(column.dropna().unique())
        if _HIGH_CARD_THRESHOLD <= len(categories):
            raise NotImplementedError()
        self.output_dtype_ = CategoricalDtype(
            categories + [f"skrub_unknown_category TODO"]
        )
        return self.transform(column)

    def transform(self, column):
        return column.astype(self.output_dtype_)
