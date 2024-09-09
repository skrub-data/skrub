from sklearn.base import BaseEstimator

from . import _dataframe as sbd
from ._exceptions import RejectColumn


class PandasConvertDTypes(BaseEstimator):
    __single_column_transformer__ = True

    def fit_transform(self, column):
        if not sbd.is_pandas(column):
            raise RejectColumn(f"Column {sbd.name(column)!r} is not a pandas Series.")
        self.original_dtype_ = sbd.dtype(column)
        column = sbd.pandas_convert_dtypes(column)
        self.target_dtype_ = sbd.dtype(column)
        return column

    def transform(self, column):
        column = sbd.pandas_convert_dtypes(column)
        return column

    def fit(self, column):
        self.fit_transform(column)
        return self
