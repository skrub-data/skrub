from sklearn.base import BaseEstimator

from . import _dataframe as sbd
from ._dataframe import asdfapi, asnative, dfapi_ns

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
    None,
]


class CleanNullStrings(BaseEstimator):
    __univariate_transformer__ = True

    def fit_transform(self, column):
        column = asdfapi(column)
        ns = dfapi_ns(column)
        if not isinstance(column.dtype, ns.String):
            raise NotImplementedError()
        is_null = column.is_in(ns.column_from_sequence(STR_NA_VALUES))
        column = sbd.where(asnative(column), ~asnative(is_null), [None])
        # TODO also replace whitespace-only values
        return column
