from collections.abc import Mapping

import pandas as pd
import pandas.api.types
from sklearn.utils.fixes import parse_version

try:
    import polars as pl
except ImportError:
    pass

from .._dispatch import dispatch

__all__ = [
    #
    # Inspecting containers' type and module
    #
    "skrub_namespace",
    "dataframe_module_name",
    "is_pandas",
    "is_polars",
    "is_dataframe",
    "is_lazyframe",
    "is_column",
    #
    # Conversions to and from other container types
    #
    "to_list",
    "to_numpy",
    "to_pandas",
    "make_dataframe_like",
    "make_column_like",
    "null_value_for",
    "all_null_like",
    "concat_horizontal",
    "to_column_list",
    "col",
    "collect",
    #
    # Querying and modifying metadata
    #
    "shape",
    "name",
    "column_names",
    "rename",
    "set_column_names",
    #
    # Inspecting dtypes and casting
    #
    "dtype",
    "cast",
    "pandas_convert_dtypes",
    "is_bool",
    "is_numeric",
    "to_numeric",
    "is_integer",
    "is_float",
    "to_float32",
    "is_string",
    "to_string",
    "is_object",
    "is_any_date",
    "to_datetime",
    "is_categorical",
    "to_categorical",
    #
    # Inspecting, selecting and modifying values
    #
    "all",
    "any",
    "is_in",
    "is_null",
    "has_nulls",
    "drop_nulls",
    "n_unique",
    "unique",
    "where",
    "sample",
    "head",
    "replace",
    "replace_regex",
]

#
# Inspecting containers' type and module
# ======================================
#


# TODO: skrub_namespace is temporary; all code in those modules should be moved
# here or in the corresponding skrub modules and use the dispatch mechanism.


@dispatch
def skrub_namespace(obj):
    """Return the skrub private module for a dataframe library.

    Returns either ``skrub._dataframe._pandas`` or ``skrub._dataframe._polars``.
    This is temporary until functions in those modules have moved elsewhere.
    """
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
    """Return the dataframe module this object belongs to: 'pandas' or 'polars'."""
    raise NotImplementedError()


@dataframe_module_name.specialize("pandas")
def _dataframe_module_name_pandas(obj):
    return "pandas"


@dataframe_module_name.specialize("polars")
def _dataframe_module_name_polars(obj):
    return "polars"


def is_pandas(obj):
    """Return True if ``obj`` is a pandas DataFrame or Series."""
    return dataframe_module_name(obj) == "pandas"


def is_polars(obj):
    """Return True if ``obj`` is a polars DataFrame, LazyFrame or Series."""
    return dataframe_module_name(obj) == "polars"


@dispatch
def is_dataframe(obj):
    """Return True if ``obj`` is a DataFrame or a LazyFrame"""
    return False


@is_dataframe.specialize("pandas", argument_type="DataFrame")
def _is_dataframe_pandas(obj):
    return True


@is_dataframe.specialize("polars", argument_type=["DataFrame", "LazyFrame"])
def _is_dataframe_polars(obj):
    return True


@dispatch
def is_lazyframe(df):
    """Return True if ``df`` is a polars LazyFrame"""
    return False


@is_lazyframe.specialize("polars", argument_type="LazyFrame")
def _is_lazyframe_polars_lazyframe(df):
    return True


@dispatch
def is_column(obj):
    """Return True if ``obj`` is a dataframe column"""
    return False


@is_column.specialize("pandas", argument_type="Column")
def _is_column_pandas(obj):
    return True


@is_column.specialize("polars", argument_type="Column")
def _is_column_polars(obj):
    return True


#
# Conversions to and from other container types
# =============================================
#


@dispatch
def to_list(col):
    raise NotImplementedError()


@to_list.specialize("pandas", argument_type="Column")
def _to_list_pandas(col):
    result = col.tolist()
    return [None if item is pd.NA else item for item in result]


@to_list.specialize("polars", argument_type="Column")
def _to_list_polars(col):
    return col.to_list()


@dispatch
def to_numpy(obj):
    raise NotImplementedError()


@to_numpy.specialize("pandas", argument_type="Column")
def _to_numpy_pandas(obj):
    if pd.api.types.is_numeric_dtype(obj) and obj.isna().any():
        obj = obj.astype(float)
    return obj.to_numpy()


@to_numpy.specialize("polars", argument_type="Column")
def _to_numpy_polars(obj):
    return obj.to_numpy()


@dispatch
def to_pandas(obj):
    raise NotImplementedError()


@to_pandas.specialize("pandas")
def _to_pandas_pandas(obj):
    return obj


@to_pandas.specialize("polars")
def _to_pandas_polars(obj):
    return obj.to_pandas().convert_dtypes()


@dispatch
def make_dataframe_like(df, data):
    """Create a dataframe from `data` using the module of `df`.

    `data` can either be a dictionary {column_name: column} or a list of columns
    (with names).

    `df` can either be a dataframe or a column, and it is only used for dispatch,
    i.e. to determine if the resulting dataframe should be a pandas or polars
    dataframe.
    """
    raise NotImplementedError()


@make_dataframe_like.specialize("pandas")
def _make_dataframe_like_pandas(df, data):
    if isinstance(data, Mapping):
        return pd.DataFrame(data)
    return pd.DataFrame({name(col): col for col in data})


@make_dataframe_like.specialize("polars")
def _make_dataframe_like_polars(df, data):
    return pl.DataFrame(data)


@dispatch
def make_column_like(column, values, name):
    raise NotImplementedError()


@make_column_like.specialize("pandas")
def _make_column_like_pandas(column, values, name):
    return pd.Series(data=values, name=name)


@make_column_like.specialize("polars")
def _make_column_like_polars(column, values, name):
    return pl.Series(values=values, name=name)


@dispatch
def null_value_for(obj):
    raise NotImplementedError()


@null_value_for.specialize("pandas")
def _null_value_for_pandas(obj):
    return pd.NA


@null_value_for.specialize("polars")
def _null_value_for_polars(obj):
    return None


@dispatch
def all_null_like(column, length=None, dtype=None, name=None):
    raise NotImplementedError()


@all_null_like.specialize("pandas")
def _all_null_like_pandas(column, length=None, dtype=None, name=None):
    if length is None:
        length = len(column)
    if dtype is None:
        dtype = column.dtype
    if name is None:
        name = column.name
    return pd.Series(dtype=dtype, index=column.index, name=name)


@all_null_like.specialize("polars")
def _all_null_like_polars(column, length=None, dtype=None, name=None):
    if length is None:
        length = len(column)
    if dtype is None:
        dtype = column.dtype
    if name is None:
        name = column.name
    return pl.Series(name, [None] * length, dtype=dtype)


@dispatch
def concat_horizontal(*dataframes):
    raise NotImplementedError()


@concat_horizontal.specialize("pandas")
def _concat_horizontal_pandas(*dataframes):
    dataframes = [df.reset_index(drop=True) for df in dataframes]
    return pd.concat(dataframes, axis=1)


@concat_horizontal.specialize("polars")
def _concat_horizontal_polars(*dataframes):
    return pl.concat(dataframes, how="horizontal")


def to_column_list(obj):
    if is_column(obj):
        return [obj]
    if is_dataframe(obj):
        return [col(obj, c) for c in column_names(obj)]
    if not hasattr(obj, "__iter__") or (len(obj) and not is_column(next(iter(obj)))):
        raise TypeError("obj should be a DataFrame, a Column or a list of Columns.")
    return obj


@dispatch
def col(df, col_name):
    raise NotImplementedError()


@col.specialize("pandas")
def _col_pandas(df, col_name):
    return df[col_name]


@col.specialize("polars")
def _col_polars(df, col_name):
    return df[col_name]


@dispatch
def collect(df):
    return df


@collect.specialize("polars", argument_type="LazyFrame")
def _collect_polars_lazyframe(df):
    return df.collect()


#
# Querying and modifying metadata
# ===============================
#


@dispatch
def shape(obj):
    raise NotImplementedError()


@shape.specialize("pandas")
def _shape_pandas(obj):
    return obj.shape


@shape.specialize("polars")
def _shape_polars(obj):
    return obj.shape


@dispatch
def name(col):
    raise NotImplementedError()


@name.specialize("pandas")
def _name_pandas(col):
    return col.name


@name.specialize("polars")
def _name_polars(col):
    return col.name


@dispatch
def column_names(df):
    raise NotImplementedError()


@column_names.specialize("pandas")
def _column_names_pandas(df):
    return list(df.columns.values)


@column_names.specialize("polars")
def _column_names_polars(df):
    return df.columns


@dispatch
def rename(col, new_name):
    raise NotImplementedError()


@rename.specialize("pandas")
def _rename_pandas(col, new_name):
    return col.rename(new_name)


@rename.specialize("polars")
def _rename_polars(col, new_name):
    return col.rename(new_name)


@dispatch
def set_column_names(df, new_column_names):
    raise NotImplementedError()


@set_column_names.specialize("pandas")
def _set_column_names_pandas(df, new_column_names):
    return df.set_axis(new_column_names, axis=1)


@set_column_names.specialize("polars")
def _set_column_names_polars(df, new_column_names):
    return df.rename(dict(zip(df.columns, new_column_names)))


#
# Inspecting dtypes and casting
# =============================
#


@dispatch
def dtype(column):
    raise NotImplementedError()


@dtype.specialize("pandas")
def _dtype_pandas(column):
    return column.dtype


@dtype.specialize("polars")
def _dtype_polars(column):
    return column.dtype


@dispatch
def cast(column, dtype):
    raise NotImplementedError()


@cast.specialize("pandas")
def _cast_pandas(column, dtype):
    return column.astype(dtype)


@cast.specialize("polars")
def _cast_polars(column, dtype):
    return column.cast(dtype)


@dispatch
def pandas_convert_dtypes(obj):
    return obj


@pandas_convert_dtypes.specialize("pandas")
def _pandas_convert_dtypes_pandas(obj):
    return obj.convert_dtypes()


@dispatch
def is_bool(column):
    raise NotImplementedError()


@is_bool.specialize("pandas")
def _is_bool_pandas(column):
    return pandas.api.types.is_bool_dtype(column)


@is_bool.specialize("polars")
def _is_bool_polars(column):
    return column.dtype == pl.Boolean


@dispatch
def is_numeric(column):
    raise NotImplementedError()


@is_numeric.specialize("pandas")
def _is_numeric_pandas(column):
    # polars and pandas disagree about whether Booleans are numbers
    return pandas.api.types.is_numeric_dtype(
        column
    ) and not pandas.api.types.is_bool_dtype(column)


@is_numeric.specialize("polars")
def _is_numeric_polars(column):
    return column.dtype.is_numeric()


@dispatch
def to_numeric(column, dtype=None, strict=True):
    raise NotImplementedError()


@to_numeric.specialize("pandas")
def _to_numeric_pandas(column, dtype=None, strict=True):
    errors = "raise" if strict else "coerce"
    out = pd.to_numeric(column, errors=errors)
    if dtype is None:
        return out.convert_dtypes()
    return out.astype(dtype)


@to_numeric.specialize("polars")
def _to_numeric_polars(column, dtype=None, strict=True):
    if dtype is not None:
        return column.cast(dtype, strict=strict)
    if column.dtype.is_numeric():
        return column
    error = None
    for dtype in [pl.Int64, pl.Float64]:
        try:
            return column.cast(dtype, strict=strict)
        except Exception as e:
            error = e
    if not strict:
        return column.cast(pl.Float64, strict=False)
    raise ValueError("Could not convert column to numeric dtype") from error


@dispatch
def is_integer(column):
    raise NotImplementedError()


@is_integer.specialize("pandas")
def _is_integer_pandas(column):
    return pd.api.types.is_integer_dtype(column)


@is_integer.specialize("polars")
def _is_integer_polars(column):
    return column.dtype.is_integer()


@dispatch
def is_float(column):
    raise NotImplementedError()


@is_float.specialize("pandas")
def _is_float_pandas(column):
    return pd.api.types.is_float_dtype(column)


@is_float.specialize("polars")
def _is_float_polars(column):
    return column.dtype.is_float()


@dispatch
def to_float32(column):
    raise NotImplementedError()


@to_float32.specialize("pandas")
def _to_float32_pandas(column):
    return _to_numeric_pandas(column, dtype=pd.Float32Dtype())


@to_float32.specialize("polars")
def _to_float32_polars(column):
    return _to_numeric_polars(column, dtype=pl.Float32)


@dispatch
def is_string(column):
    raise NotImplementedError()


@is_string.specialize("pandas")
def _is_string_pandas(column):
    if parse_version(pd.__version__) < parse_version("2.0.0"):
        # on old pandas versions
        # `pd.api.types.is_string_dtype(pd.Series([1, ""]))` is True
        return column.convert_dtypes().dtype == pd.StringDtype()
    return pandas.api.types.is_string_dtype(column) and not isinstance(
        column.dtype, pandas.CategoricalDtype
    )


@is_string.specialize("polars")
def _is_string_polars(column):
    return column.dtype == pl.String


@dispatch
def to_string(column):
    raise NotImplementedError()


@to_string.specialize("pandas")
def _to_string_pandas(column):
    return column.astype(pd.StringDtype())


@to_string.specialize("polars")
def _to_string_polars(column):
    return column.cast(pl.String)


@dispatch
def is_object(column):
    raise NotImplementedError()


@is_object.specialize("pandas")
def _is_object_pandas(column):
    return pandas.api.types.is_object_dtype(column)


@is_object.specialize("polars")
def _is_object_polars(column):
    return column.dtype == pl.Object


@dispatch
def is_any_date(column):
    raise NotImplementedError()


@is_any_date.specialize("pandas")
def _is_any_date_pandas(column):
    return pandas.api.types.is_datetime64_any_dtype(column)


@is_any_date.specialize("polars")
def _is_any_date_polars(column):
    return column.dtype in (pl.Date, pl.Datetime)


@dispatch
def to_datetime(column, format, strict=True):
    raise NotImplementedError()


@to_datetime.specialize("pandas")
def _to_datetime_pandas(column, format, strict=True):
    if _is_any_date_pandas(column):
        return column
    errors = "raise" if strict else "coerce"
    out = pd.to_datetime(column, format=format, errors=errors)
    if out.dt.tz is not None:
        out = out.dt.tz_convert("UTC")
    return out


@to_datetime.specialize("polars")
def _to_datetime_polars(column, format, strict=True):
    if _is_any_date_polars(column):
        return column
    try:
        return column.str.to_datetime(format=format, strict=strict)
    except pl.ComputeError as e:
        raise ValueError("Failed to convert to datetime") from e


@dispatch
def is_categorical(column):
    raise NotImplementedError()


@is_categorical.specialize("pandas")
def _is_categorical_pandas(column):
    return isinstance(column.dtype, pd.CategoricalDtype)


@is_categorical.specialize("polars")
def _is_categorical_polars(column):
    return column.dtype in (pl.Categorical, pl.Enum)


@dispatch
def to_categorical(column):
    raise NotImplementedError()


@to_categorical.specialize("pandas")
def _to_categorical_pandas(column):
    return column.astype("category")


@to_categorical.specialize("polars")
def _to_categorical_polars(column):
    return column.cast(pl.Categorical())


#
# Inspecting, selecting and modifying values
# ==========================================
#


@dispatch
def all(column):
    raise NotImplementedError()


@all.specialize("pandas")
def _all_pandas(column):
    return column.all()


@all.specialize("polars")
def _all_polars(column):
    return column.all()


@dispatch
def any(column):
    raise NotImplementedError()


@any.specialize("pandas")
def _any_pandas(column):
    return column.any()


@any.specialize("polars")
def _any_polars(column):
    return column.any()


@dispatch
def is_in(column, values):
    raise NotImplementedError()


@is_in.specialize("pandas")
def _is_in_pandas(column, values):
    res = column.isin(values).convert_dtypes()
    res[column.isna()] = pd.NA
    return res


@is_in.specialize("polars")
def _is_in_polars(column, values):
    return column.is_in(values)


@dispatch
def is_null(column):
    raise NotImplementedError()


@is_null.specialize("pandas")
def _is_null_pandas(column):
    return rename(column.isna(), name(column))


@is_null.specialize("polars")
def _is_null_polars(column):
    return column.is_null()


def has_nulls(column):
    return any(is_null(column))


@dispatch
def drop_nulls(column):
    raise NotImplementedError()


@drop_nulls.specialize("pandas")
def _drop_nulls_pandas(column):
    return column.dropna().reset_index(drop=True)


@drop_nulls.specialize("polars")
def _drop_nulls_polars(column):
    return column.drop_nulls()


@dispatch
def n_unique(column):
    raise NotImplementedError()


@n_unique.specialize("pandas")
def _n_unique_pandas(column):
    return column.nunique()


@n_unique.specialize("polars")
def _n_unique_polars(column):
    n = column.n_unique()
    if column.is_null().any():
        n -= 1
    return n


@dispatch
def unique(column):
    raise NotImplementedError()


@unique.specialize("pandas")
def _unique_pandas(column):
    return pd.Series(column.dropna().unique()).rename(column.name)


@unique.specialize("polars")
def _unique_polars(column):
    return column.unique().drop_nulls()


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


@dispatch
def head(df, n=5):
    raise NotImplementedError()


@head.specialize("pandas")
def _head_pandas(df, n=5):
    return df.head(n=n)


@head.specialize("polars")
def _head_polars(df, n=5):
    return df.head(n=n)


@dispatch
def replace(column, old, new):
    raise NotImplementedError()


@replace.specialize("pandas")
def _replace_pandas(column, old, new):
    return column.replace(old, new, regex=False)


@replace.specialize("polars")
def _replace_polars(column, old, new):
    return column.replace(old, new)


@dispatch
def replace_regex(column, pattern, replacement):
    # NOTE: regex syntax is not the same in python and polars in particular for
    # referring to capture groups. ATM we don't deal with that ie capture
    # groups will not work.
    raise NotImplementedError()


@replace_regex.specialize("pandas")
def _replace_regex_pandas(column, pattern, replacement):
    return column.str.replace(pattern, replacement, regex=True)


@replace_regex.specialize("polars")
def _replace_regex_polars(column, pattern, replacement):
    return column.str.replace_all(pattern, replacement, literal=False)
