from sklearn.base import BaseEstimator

from . import _dataframe as sbd
from ._exceptions import RejectColumn

# Taken from pandas.io.parsers (version 1.1.4)
STR_NA_VALUES = [
    "null",
    "",
    "1.#QNAN",
    "#NA",
    "nan",
    "#N/A N/A",
    "-1.#QNAN",
    "<NA>",
    "-1.#IND",
    "-nan",
    "n/a",
    "-NaN",
    "1.#IND",
    "NULL",
    "NA",
    "N/A",
    "#N/A",
    "NaN",
    "?",
    "...",
]


class CleanNullStrings(BaseEstimator):
    __single_column_transformer__ = True

    def fit_transform(self, column):
        if not sbd.is_string(column) or sbd.is_categorical(column):
            raise RejectColumn(f"Column {sbd.name(column)!r} does not contain strings.")
        return self.transform(column)

    def transform(self, column):
        column = sbd.replace_regex(column, r"^\s*$", "")
        column = sbd.replace(column, STR_NA_VALUES, sbd.null_value_for(column))
        return column

    def fit(self, column):
        self.fit_transform(column)
        return self
