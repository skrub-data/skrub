"""This transformer converts durations to seconds."""

from . import _dataframe as sbd
from ._single_column_transformer import RejectColumn, SingleColumnTransformer


class ConvertDuration(SingleColumnTransformer):
    def fit_transform(self, col, y=None):
        del y
        if not sbd.is_duration(col):
            raise RejectColumn(f"Expected a duration column, got {col.dtype}")
        return self.transform(col)

    def transform(self, col, y=None):
        del y

        column = sbd.convert_duration(col)
        return column
