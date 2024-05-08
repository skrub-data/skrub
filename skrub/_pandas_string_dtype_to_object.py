import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from . import _dataframe as sbd
from ._exceptions import RejectColumn


class PandasStringDtypeToObject(BaseEstimator):
    __single_column_transformer__ = True

    def fit_transform(self, column):
        if not sbd.is_pandas(column):
            raise RejectColumn("Column {sbd.name(column)!r} is not a pandas Series.")
        if not sbd.is_string(column):
            raise RejectColumn(
                "Column {sbd.name(column)!r} does not have a str or string dtype."
            )
        return self.transform(column)

    def transform(self, column):
        assert sbd.is_pandas(column)
        if column.dtype == pd.StringDtype():
            return column.astype("str")
        if not column.isna().any():
            return column
        return column.fillna(np.nan)
