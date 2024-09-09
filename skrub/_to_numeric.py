from sklearn.base import BaseEstimator

from . import _dataframe as sbd
from ._exceptions import RejectColumn


class ToNumeric(BaseEstimator):
    __single_column_transformer__ = True

    def fit_transform(self, column):
        if sbd.is_any_date(column) or sbd.is_categorical(column) or sbd.is_bool(column):
            raise RejectColumn(
                f"Refusing to cast column {sbd.name(column)!r} "
                f"with dtype {sbd.dtype(column)} to numbers."
            )
        try:
            numeric = sbd.to_numeric(column)
            self.output_dtype_ = sbd.dtype(numeric)
            return numeric
        except Exception as e:
            raise RejectColumn(
                f"Could not convert column {sbd.name(column)!r} to numbers."
            ) from e

    def transform(self, column):
        return sbd.to_numeric(column, dtype=self.output_dtype_, strict=False)

    def fit(self, column):
        self.fit_transform(column)
        return self
