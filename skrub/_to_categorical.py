try:
    import pandas as pd
except ImportError:
    pass
try:
    import polars as pl
except ImportError:
    pass

from sklearn.base import BaseEstimator

from . import _dataframe as sbd
from . import _utils
from ._dispatch import dispatch
from ._exceptions import RejectColumn


@dispatch
def _is_enum(column):
    raise NotImplementedError()


@_is_enum.specialize("pandas")
def _is_enum_pandas(column):
    return isinstance(column.dtype, pd.CategoricalDtype)


@_is_enum.specialize("polars")
def _is_enum_polars(column):
    return column.dtype == pl.Enum


@dispatch
def _make_enum_dtype_for(obj, categories):
    raise NotImplementedError()


@_make_enum_dtype_for.specialize("pandas")
def _make_enum_dtype_for_pandas(obj, categories):
    return pd.CategoricalDtype(categories)


@_make_enum_dtype_for.specialize("polars")
def _make_enum_dtype_for_polars(obj, categories):
    return pl.Enum(categories)


@dispatch
def _dtype_categories(column):
    raise NotImplementedError()


@_dtype_categories.specialize("pandas")
def _dtype_categories_pandas(column):
    return list(column.dtype.categories)


@_dtype_categories.specialize("polars")
def _dtype_categories_polars(column):
    return column.dtype.categories


class ToCategorical(BaseEstimator):
    __single_column_transformer__ = True

    def __init__(self, max_categories=40):
        self.max_categories = max_categories

    def fit_transform(self, column):
        if _is_enum(column):
            self.output_dtype_ = sbd.dtype(column)
            self._categories = _dtype_categories(column)
            self.unknown_category_ = None
            return column
        if not (sbd.is_string(column) or sbd.is_categorical(column)):
            raise RejectColumn(
                f"Column {sbd.name(column)!r} does not contain strings or categories."
            )
        categories = list(sbd.drop_nulls(sbd.unique(column)))
        if self.max_categories <= len(categories):
            raise RejectColumn(
                f"Cardinality of column {sbd.name(column)!r} "
                f"is >= threshold {self.max_categories}."
            )
        token = _utils.random_string()
        self.unknown_category_ = f"skrub_unknown_category_{token}"
        self._categories = categories + [self.unknown_category_]
        self.output_dtype_ = _make_enum_dtype_for(column, self._categories)
        return self.transform(column)

    def transform(self, column):
        if sbd.dtype(column) == self.output_dtype_:
            return column
        keep = sbd.is_in(column, self._categories) | sbd.is_null(column)
        column = sbd.where(column, keep, [self.unknown_category_])
        return sbd.cast(column, self.output_dtype_)

    def fit(self, column):
        self.fit_transform(column)
        return self
