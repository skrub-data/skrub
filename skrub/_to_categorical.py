import pandas as pd
from sklearn.base import BaseEstimator

from . import _dataframe as sbd
from ._dispatch import dispatch
from ._exceptions import RejectColumn


@dispatch
def _with_string_categories(column):
    raise NotImplementedError()


@_with_string_categories.specialize("pandas")
def _with_string_categories_pandas(column):
    categories = column.cat.categories.to_series()
    if pd.api.types.is_string_dtype(
        categories
    ) and not pd.api.types.is_extension_array_dtype(categories):
        return column
    try:
        return column.cat.rename_categories(categories.astype("str"))
    except ValueError:
        # unlikely case that different values in categories have the same
        # string representation: recompute unique categories after casting to
        # string
        return column.astype("str").astype("category")


@_with_string_categories.specialize("polars")
def _with_string_categories_polars(column):
    return column


class ToCategorical(BaseEstimator):
    __single_column_transformer__ = True

    def __init__(self, max_categories=None):
        self.max_categories = max_categories

    def fit_transform(self, column):
        if sbd.is_categorical(column):
            return _with_string_categories(column)
        if not sbd.is_string(column):
            raise RejectColumn(
                f"Column {sbd.name(column)!r} does not contain strings or categories."
            )
        n_categories = len(sbd.drop_nulls(sbd.unique(column)))
        if self.max_categories is not None and self.max_categories < n_categories:
            raise RejectColumn(
                f"Cardinality of column {sbd.name(column)!r} "
                f"is > max_categories, which is: {self.max_categories}."
            )
        return _with_string_categories(sbd.to_categorical(column))

    def transform(self, column):
        return _with_string_categories(sbd.to_categorical(column))

    def fit(self, column):
        self.fit_transform(column)
        return self
