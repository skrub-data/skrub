try:
    import pandas as pd
except ImportError:
    pass
try:
    import polars as pl
except ImportError:
    pass

from ._dataframe_api import asdfapi, asnative, dfapi_ns
from ._dispatch import dispatch

__all__ = [
    "skrub_namespace",
    "dataframe_module_name",
    "shape",
    "is_numeric",
    "is_temporal",
    "numeric_column_names",
    "temporal_column_names",
    "select",
    "set_column_names",
    "collect",
    "is_categorical",
    "to_categorical",
    "make_categorical_dtype_for",
    "to_datetime",
    "unique",
    "native_dtype",
    "native_cast",
    "where",
    "sample",
]


# TODO: skrub_namespace is temporary; all code in those modules should be moved
# here or in the corresponding skrub modules and use the dispatch mechanism.
@dispatch
def skrub_namespace(obj):
    raise NotImplementedError()


@skrub_namespace.specialize("pandas")
def _skrub_namespace_pandas(obj):
    from . import _pandas

    return _pandas


@skrub_namespace.specialize("polars")
def _skrub_namespace_polars(obj):
    from . import _polars

    return _polars


@dispatch
def dataframe_module_name(obj):
    return None


@dataframe_module_name.specialize("pandas")
def _dataframe_module_name_pandas(obj):
    return "pandas"


@dataframe_module_name.specialize("polars")
def _dataframe_module_name_polars(obj):
    return "polars"


def shape(obj):
    obj = asdfapi(obj)
    if hasattr(obj, "shape"):
        try:
            shape = obj.shape()
        except ValueError:
            shape = obj.persist().shape()
    else:
        assert hasattr(obj, "len")
        shape = (obj.len(),)
    return tuple(map(asnative, shape))


def is_numeric(column, include_bool=True):
    column = asdfapi(column)
    ns = dfapi_ns(column)
    if ns.is_dtype(column.dtype, "numeric"):
        return True
    if not include_bool:
        return False
    return ns.is_dtype(column.dtype, "bool")


def is_temporal(column):
    column = asdfapi(column)
    ns = dfapi_ns(column)
    for dtype in [ns.Date, ns.Datetime]:
        if isinstance(column.dtype, dtype):
            return True
    return False


def _select_column_names(df, predicate):
    df = asdfapi(df)
    return [col_name for col_name in df.column_names if predicate(df.col(col_name))]


def numeric_column_names(df):
    return _select_column_names(df, is_numeric)


def temporal_column_names(df):
    return _select_column_names(df, is_temporal)


def select(df, column_names):
    return asnative(asdfapi(df).select(*column_names))


@dispatch
def collect(df):
    return df


@collect.specialize("polars", "LazyFrame")
def _collect_polars_lazyframe(df):
    return df.collect()


def set_column_names(df, new_column_names):
    df = asdfapi(df)
    ns = dfapi_ns(df)
    new_columns = (
        df.col(col_name).rename(new_name).persist()
        for (col_name, new_name) in zip(df.column_names, new_column_names)
    )
    return asnative(ns.dataframe_from_columns(*new_columns))


@dispatch
def is_categorical(column):
    raise NotImplementedError()


@is_categorical.specialize("pandas")
def _is_categorical_pandas(column):
    return isinstance(column.dtype, pd.CategoricalDtype)


@is_categorical.specialize("polars")
def _is_categorical_polars(column):
    return isinstance(column.dtype, pl.Categorical)


@dispatch
def to_categorical(column):
    raise NotImplementedError()


@to_categorical.specialize("pandas")
def _to_categorical_pandas(column):
    return column.astype("category")


@to_categorical.specialize("polars")
def _to_categorical_polars(column):
    return column.cast(pl.Categorical())


@dispatch
def make_categorical_dtype_for(obj, categories):
    raise NotImplementedError()


@make_categorical_dtype_for.specialize("pandas")
def _make_categorical_dtype_for_pandas(obj, categories):
    return pd.CategoricalDtype(categories)


@make_categorical_dtype_for.specialize("polars")
def _make_categorical_dtype_for_polars(obj, categories):
    return pl.Enum(categories)


@dispatch
def to_datetime(column, format):
    raise NotImplementedError()


@to_datetime.specialize("pandas")
def _to_datetime_pandas(column, format):
    return pd.to_datetime(column, format=format)


@to_datetime.specialize("polars")
def _to_datetime_polars(column, format):
    return column.str.to_datetime(format=format)


@dispatch
def unique(column):
    column = asdfapi(column)
    unique_values = column.take(column.unique_indices(skip_nulls=False))
    return asnative(unique_values)


@unique.specialize("pandas")
def _unique_pandas(column):
    return pd.Series(column.dropna().unique())


@unique.specialize("polars")
def _unique_polars(column):
    return column.unique().drop_nulls()


@dispatch
def native_dtype(column):
    return column.dtype


@native_dtype.specialize("pandas")
def _native_dtype_pandas(column):
    return column.dtype


@native_dtype.specialize("polars")
def _native_dtype_polars(column):
    return column.dtype


@dispatch
def native_cast(column, dtype):
    raise NotImplementedError()


@native_cast.specialize("pandas")
def _native_cast_pandas(column, dtype):
    return column.astype(dtype)


@native_cast.specialize("polars")
def _native_cast_polars(column, dtype):
    return column.cast(dtype)


@dispatch
def where(column, mask, other):
    raise NotImplementedError()


@where.specialize("pandas")
def _where_pandas(column, mask, other):
    return column.where(mask, pd.Series(other))


@where.specialize("polars")
def _where_polars(column, mask, other):
    return column.zip_with(mask, pl.Series(other))


@dispatch
def sample(obj, n, seed=None):
    raise NotImplementedError()


@sample.specialize("pandas")
def _sample_pandas(obj, n, seed=None):
    return obj.sample(n=n, random_state=seed, replace=False)


@sample.specialize("polars")
def _sample_polars(obj, n, seed=None):
    return obj.sample(n=n, seed=seed, with_replacement=False)
