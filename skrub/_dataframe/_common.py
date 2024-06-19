from collections.abc import Mapping, Sequence

import numpy as np
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
    "is_column_list",
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
    "reset_index",
    "copy_index",
    "index",
    #
    # Inspecting dtypes and casting
    #
    "dtype",
    "dtypes",
    "cast",
    "is_pandas_extension_dtype",
    "pandas_convert_dtypes",
    "is_bool",
    "is_numeric",
    "is_integer",
    "is_float",
    "to_float32",
    "is_string",
    "to_string",
    "is_object",
    "is_pandas_object",
    "is_any_date",
    "to_datetime",
    "is_categorical",
    "to_categorical",
    #
    # Inspecting, selecting and modifying values
    #
    "all",
    "any",
    "is_null",
    "has_nulls",
    "drop_nulls",
    "fill_nulls",
    "n_unique",
    "unique",
    "filter",
    "where",
    "sample",
    "head",
    "replace",
    "with_columns",
]

#
# Inspecting containers' type and module
# ======================================
#


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
    """Return True if ``obj`` is a DataFrame or a LazyFrame."""
    return False


@is_dataframe.specialize("pandas", argument_type="DataFrame")
def _is_dataframe_pandas(obj):
    return True


@is_dataframe.specialize("polars", argument_type="DataFrame")
def _is_dataframe_polars(obj):
    return True


@dispatch
def is_lazyframe(obj):
    """Return True if ``obj`` is a polars LazyFrame."""
    return False


@is_lazyframe.specialize("polars", argument_type="LazyFrame")
def _is_lazyframe_polars_lazyframe(obj):
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
def to_numpy(col):
    raise NotImplementedError()


@to_numpy.specialize("pandas", argument_type="Column")
def _to_numpy_pandas_column(col):
    if pd.api.types.is_numeric_dtype(col) and col.isna().any():
        col = col.astype(float)
    return col.to_numpy()


@to_numpy.specialize("polars", argument_type="Column")
def _to_numpy_polars_column(col):
    return col.to_numpy()


@dispatch
def to_pandas(obj):
    raise NotImplementedError()


@to_pandas.specialize("pandas")
def _to_pandas_pandas(obj):
    return obj


@to_pandas.specialize("polars")
def _to_pandas_polars(obj):
    return obj.to_pandas()


@dispatch
def make_dataframe_like(obj, data):
    """Create a dataframe from `data` using the module of `obj`.

    `data` can either be a dictionary {column_name: column} or a list of columns
    (with names).

    `obj` can either be a dataframe or a column, and it is only used for dispatch,
    i.e. to determine if the resulting dataframe should be a pandas or polars
    dataframe.
    """
    raise NotImplementedError()


@make_dataframe_like.specialize("pandas")
def _make_dataframe_like_pandas(obj, data):
    if isinstance(data, Mapping):
        return pd.DataFrame({k: reset_index(v) for k, v in data.items()}, copy=False)
    return pd.DataFrame({name(col): reset_index(col) for col in data}, copy=False)


@make_dataframe_like.specialize("polars")
def _make_dataframe_like_polars(obj, data):
    return pl.DataFrame(data)


@dispatch
def make_column_like(obj, values, name):
    raise NotImplementedError()


@make_column_like.specialize("pandas")
def _make_column_like_pandas(obj, values, name):
    return pd.Series(data=values, name=name)


@make_column_like.specialize("polars")
def _make_column_like_polars(obj, values, name):
    return pl.Series(values=values, name=name)


@dispatch
def null_value_for(obj):
    raise NotImplementedError()


@null_value_for.specialize("pandas")
def _null_value_for_pandas(obj):
    if pd.api.types.is_extension_array_dtype(obj):
        return pd.NA
    return None


@null_value_for.specialize("polars")
def _null_value_for_polars(obj):
    return None


@dispatch
def all_null_like(col, length=None, dtype=None, name=None):
    raise NotImplementedError()


@all_null_like.specialize("pandas", argument_type="Column")
def _all_null_like_pandas(col, length=None, dtype=None, name=None):
    if length is None:
        length = len(col)
    if dtype is None:
        dtype = col.dtype
    if name is None:
        name = col.name
    return pd.Series(dtype=dtype, index=col.index, name=name)


@all_null_like.specialize("polars", argument_type="Column")
def _all_null_like_polars(col, length=None, dtype=None, name=None):
    if length is None:
        length = len(col)
    if dtype is None:
        dtype = col.dtype
    if name is None:
        name = col.name
    return pl.Series(name, [None] * length, dtype=dtype)


@dispatch
def concat_horizontal(*dataframes):
    raise NotImplementedError()


@concat_horizontal.specialize("pandas")
def _concat_horizontal_pandas(*dataframes):
    dataframes = [df.reset_index(drop=True) for df in dataframes]
    return pd.concat(dataframes, axis=1, copy=False)


@concat_horizontal.specialize("polars")
def _concat_horizontal_polars(*dataframes):
    return pl.concat(dataframes, how="horizontal")


def is_column_list(obj):
    if not isinstance(obj, Sequence):
        return False
    if not len(obj):
        return True
    if is_column(obj[0]):
        return True
    return False


def to_column_list(obj):
    if is_column(obj):
        return [obj]
    if is_dataframe(obj):
        return [col(obj, c) for c in column_names(obj)]
    if not is_column_list(obj):
        raise TypeError("obj should be a DataFrame, a Column or a list of Columns.")
    return obj


@dispatch
def col(df, col_name):
    raise NotImplementedError()


@col.specialize("pandas", argument_type="DataFrame")
def _col_pandas(df, col_name):
    return df[col_name]


@col.specialize("polars", argument_type="DataFrame")
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


@name.specialize("pandas", argument_type="Column")
def _name_pandas(col):
    return col.name


@name.specialize("polars", argument_type="Column")
def _name_polars(col):
    return col.name


@dispatch
def column_names(df):
    raise NotImplementedError()


@column_names.specialize("pandas", argument_type="DataFrame")
def _column_names_pandas(df):
    return list(df.columns.values)


@column_names.specialize("polars", argument_type="DataFrame")
def _column_names_polars(df):
    return df.columns


@dispatch
def rename(col, new_name):
    raise NotImplementedError()


@rename.specialize("pandas", argument_type="Column")
def _rename_pandas(col, new_name):
    return col.rename(new_name)


@rename.specialize("polars", argument_type="Column")
def _rename_polars(col, new_name):
    return col.rename(new_name)


@dispatch
def set_column_names(df, new_col_names):
    raise NotImplementedError()


@set_column_names.specialize("pandas", argument_type="DataFrame")
def _set_column_names_pandas(df, new_col_names):
    return df.set_axis(new_col_names, axis=1)


@set_column_names.specialize("polars", argument_type="DataFrame")
def _set_column_names_polars(df, new_col_names):
    return df.rename(dict(zip(df.columns, new_col_names)))


@dispatch
def reset_index(obj):
    """Reset the index of a pandas dataframe or series.

    If ``obj`` is a pandas dataframe or series, returns ``obj.reset_index(drop=True)``.
    Otherwise this is a no-op: it returns ``obj`` itself without modifying it.
    """
    return obj


@reset_index.specialize("pandas")
def _reset_index_pandas(obj):
    return obj.reset_index(drop=True)


@dispatch
def _set_index(obj, index):
    return obj


@_set_index.specialize("pandas")
def _set_index_pandas(obj, index):
    if index is None:
        return obj
    return obj.set_axis(index, axis="index")


def copy_index(source, target):
    """Copy index from source to target.

    If both are pandas dataframes or series, returns a new object which is
    identical to `target` but with the index of `source`. `target` itself is
    not modified.

    If either is not a pandas object, return `target` itself (unchanged).
    """
    return _set_index(target, index(source))


@dispatch
def index(obj):
    return None


@index.specialize("pandas")
def _index_pandas(obj):
    return obj.index


#
# Inspecting dtypes and casting
# =============================
#


@dispatch
def dtype(col):
    raise NotImplementedError()


@dtype.specialize("pandas", argument_type="Column")
def _dtype_pandas(col):
    return col.dtype


@dtype.specialize("polars", argument_type="Column")
def _dtype_polars(col):
    return col.dtype


@dispatch
def dtypes(df):
    raise NotImplementedError()


@dtypes.specialize("pandas", argument_type="DataFrame")
def _dtypes_pandas(df):
    return list(df.dtypes.values)


@dtypes.specialize("polars", argument_type="DataFrame")
def _dtypes_polars(df):
    return df.dtypes


@dispatch
def cast(col, dtype):
    raise NotImplementedError()


@cast.specialize("pandas", argument_type="Column")
def _cast_pandas(col, dtype):
    if col.dtype == dtype:
        return col
    return col.astype(dtype)


@cast.specialize("polars", argument_type="Column")
def _cast_polars(col, dtype):
    if col.dtype == dtype:
        return col
    return col.cast(dtype)


@dispatch
def is_pandas_extension_dtype(obj):
    raise NotImplementedError()


@is_pandas_extension_dtype.specialize("pandas")
def _is_pandas_extension_dtype_pandas(obj):
    return pd.api.types.is_extension_array_dtype(obj)


@is_pandas_extension_dtype.specialize("polars")
def _is_pandas_extension_dtype_polars(obj):
    return False


@dispatch
def pandas_convert_dtypes(obj):
    return obj


@pandas_convert_dtypes.specialize("pandas")
def _pandas_convert_dtypes_pandas(obj):
    return obj.convert_dtypes()


@dispatch
def is_bool(col):
    raise NotImplementedError()


@is_bool.specialize("pandas", argument_type="Column")
def _is_bool_pandas(col):
    if pd.api.types.is_object_dtype(col):
        return pandas.api.types.is_bool_dtype(col.convert_dtypes())
    return pandas.api.types.is_bool_dtype(col)


@is_bool.specialize("polars", argument_type="Column")
def _is_bool_polars(col):
    return col.dtype == pl.Boolean


@dispatch
def is_numeric(col):
    raise NotImplementedError()


@is_numeric.specialize("pandas", argument_type="Column")
def _is_numeric_pandas(col):
    # polars and pandas disagree about whether Booleans are numbers
    return pandas.api.types.is_numeric_dtype(
        col
    ) and not pandas.api.types.is_bool_dtype(col)


@is_numeric.specialize("polars", argument_type="Column")
def _is_numeric_polars(col):
    return col.dtype.is_numeric()


@dispatch
def is_integer(col):
    raise NotImplementedError()


@is_integer.specialize("pandas", argument_type="Column")
def _is_integer_pandas(col):
    return pd.api.types.is_integer_dtype(col)


@is_integer.specialize("polars", argument_type="Column")
def _is_integer_polars(col):
    return col.dtype.is_integer()


@dispatch
def is_float(col):
    raise NotImplementedError()


@is_float.specialize("pandas", argument_type="Column")
def _is_float_pandas(col):
    return pd.api.types.is_float_dtype(col)


@is_float.specialize("polars", argument_type="Column")
def _is_float_polars(col):
    return col.dtype.is_float()


@dispatch
def to_float32(col, strict=True):
    raise NotImplementedError()


@to_float32.specialize("pandas", argument_type="Column")
def _to_float32_pandas(col, strict=True):
    if not pd.api.types.is_numeric_dtype(col):
        col = pd.to_numeric(col, errors="raise" if strict else "coerce")
    if col.dtype == np.float32:
        return col
    return col.astype(np.float32)


@to_float32.specialize("polars", argument_type="Column")
def _to_float32_polars(col, strict=True):
    if col.dtype == pl.Float32:
        return col
    return col.cast(pl.Float32, strict=strict)


@dispatch
def is_string(col):
    raise NotImplementedError()


@is_string.specialize("pandas", argument_type="Column")
def _is_string_pandas(col):
    if col.dtype == pd.StringDtype():
        return True
    if not pd.api.types.is_object_dtype(col):
        return False
    if parse_version(pd.__version__) < parse_version("2.0.0"):
        # on old pandas versions
        # `pd.api.types.is_string_dtype(pd.Series([1, ""]))` is True
        return col.convert_dtypes().dtype == pd.StringDtype()
    return pandas.api.types.is_string_dtype(col[~col.isna()])


@is_string.specialize("polars", argument_type="Column")
def _is_string_polars(col):
    return col.dtype == pl.String


@dispatch
def to_string(col):
    raise NotImplementedError()


@to_string.specialize("pandas", argument_type="Column")
def _to_string_pandas(col):
    is_na = col.isna()
    if not (is_pandas_object(col) and is_string(col)):
        col = col.astype("str")
        col[is_na] = np.nan
        return col
    if not is_na.any():
        return col
    return col.fillna(np.nan)


@to_string.specialize("polars", argument_type="Column")
def _to_string_polars(col):
    if col.dtype != pl.Object:
        # Objects are mere passengers in polars dataframes and we can't do
        # anything with them; map_elements raises an exception.
        return _cast_polars(col, pl.String)
    return col.map_elements(str)


@dispatch
def is_object(col):
    raise NotImplementedError()


@is_object.specialize("pandas", argument_type="Column")
def _is_object_pandas(col):
    return pandas.api.types.is_object_dtype(col)


@is_object.specialize("polars", argument_type="Column")
def _is_object_polars(col):
    return col.dtype == pl.Object


def is_pandas_object(col):
    return is_pandas(col) and is_object(col)


@dispatch
def is_any_date(col):
    raise NotImplementedError()


@is_any_date.specialize("pandas", argument_type="Column")
def _is_any_date_pandas(col):
    return pandas.api.types.is_datetime64_any_dtype(col)


@is_any_date.specialize("polars", argument_type="Column")
def _is_any_date_polars(col):
    return col.dtype in (pl.Date, pl.Datetime)


@dispatch
def to_datetime(col, format, strict=True):
    raise NotImplementedError()


@to_datetime.specialize("pandas", argument_type="Column")
def _to_datetime_pandas(col, format, strict=True):
    if _is_any_date_pandas(col):
        return col
    errors = "raise" if strict else "coerce"
    utc = (format is None) or ("%z" in format)
    return pd.to_datetime(col, format=format, errors=errors, utc=utc)


@to_datetime.specialize("polars", argument_type="Column")
def _to_datetime_polars(col, format, strict=True):
    if _is_any_date_polars(col):
        return col
    try:
        # avoid ChronoFormatWarning due to pandas and polars writing this
        # differently.
        format = format.replace(".%f", "%.f")
        return col.str.to_datetime(format=format, strict=strict)
    except pl.ComputeError as e:
        raise ValueError("Failed to convert to datetime") from e


@dispatch
def is_categorical(col):
    raise NotImplementedError()


@is_categorical.specialize("pandas", argument_type="Column")
def _is_categorical_pandas(col):
    return isinstance(col.dtype, pd.CategoricalDtype)


@is_categorical.specialize("polars", argument_type="Column")
def _is_categorical_polars(col):
    return col.dtype in (pl.Categorical, pl.Enum)


@dispatch
def to_categorical(col):
    raise NotImplementedError()


@to_categorical.specialize("pandas", argument_type="Column")
def _to_categorical_pandas(col):
    return _cast_pandas(col, "category")


@to_categorical.specialize("polars", argument_type="Column")
def _to_categorical_polars(col):
    if col.dtype in (pl.Categorical, pl.Enum):
        return col
    col = to_string(col)
    return _cast_polars(col, pl.Categorical())


#
# Inspecting, selecting and modifying values
# ==========================================
#


@dispatch
def all(col):
    raise NotImplementedError()


@all.specialize("pandas", argument_type="Column")
def _all_pandas(col):
    return col.all()


@all.specialize("polars", argument_type="Column")
def _all_polars(col):
    return col.all()


@dispatch
def any(col):
    raise NotImplementedError()


@any.specialize("pandas", argument_type="Column")
def _any_pandas(col):
    return col.any()


@any.specialize("polars", argument_type="Column")
def _any_polars(col):
    return col.any()


@dispatch
def is_null(col):
    raise NotImplementedError()


@is_null.specialize("pandas", argument_type="Column")
def _is_null_pandas(col):
    return rename(col.isna(), name(col))


@is_null.specialize("polars", argument_type="Column")
def _is_null_polars(col):
    if col.dtype.is_float():
        return col.is_null() | col.is_nan()
    return col.is_null()


def has_nulls(col):
    return any(is_null(col))


@dispatch
def drop_nulls(col):
    raise NotImplementedError()


@drop_nulls.specialize("pandas", argument_type="Column")
def _drop_nulls_pandas(col):
    return col.dropna().reset_index(drop=True)


@drop_nulls.specialize("polars", argument_type="Column")
def _drop_nulls_polars(col):
    if col.dtype.is_float():
        return col.drop_nulls().drop_nans()
    return col.drop_nulls()


@dispatch
def fill_nulls(obj, value):
    raise NotImplementedError()


@fill_nulls.specialize("pandas")
def _fill_nulls_pandas(obj, value):
    return obj.fillna(value)


@fill_nulls.specialize("polars", argument_type="Column")
def _fill_nulls_polars_col(col, value):
    if col.dtype.is_float():
        return col.fill_nan(value).fill_null(value)
    return col.fill_null(value)


@fill_nulls.specialize("polars", argument_type="DataFrame")
def _fill_nulls_polars_dataframe(df, value):
    from polars import selectors as cs

    return df.with_columns(
        cs.float().fill_nan(value).fill_null(value), (~cs.float()).fill_null(value)
    )


@dispatch
def n_unique(col):
    raise NotImplementedError()


@n_unique.specialize("pandas", argument_type="Column")
def _n_unique_pandas(col):
    return col.nunique()


@n_unique.specialize("polars", argument_type="Column")
def _n_unique_polars(col):
    n = col.n_unique()
    if col.is_null().any():
        n -= 1
    return n


@dispatch
def unique(col):
    raise NotImplementedError()


@unique.specialize("pandas", argument_type="Column")
def _unique_pandas(col):
    return pd.Series(col.dropna().unique()).rename(col.name)


@unique.specialize("polars", argument_type="Column")
def _unique_polars(col):
    return col.unique().drop_nulls()


@dispatch
def filter(obj, predicate):
    raise NotImplementedError()


@filter.specialize("pandas")
def _filter_pandas(obj, predicate):
    return obj[predicate]


@filter.specialize("polars")
def _filter_polars(obj, predicate):
    return obj.filter(predicate)


@dispatch
def where(col, mask, other):
    raise NotImplementedError()


@where.specialize("pandas", argument_type="Column")
def _where_pandas(col, mask, other):
    return col.where(mask, pd.Series(other))


@where.specialize("polars", argument_type="Column")
def _where_polars(col, mask, other):
    return col.zip_with(mask, pl.Series(other))


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
def head(obj, n=5):
    raise NotImplementedError()


@head.specialize("pandas")
def _head_pandas(obj, n=5):
    return obj.head(n=n)


@head.specialize("polars")
def _head_polars(obj, n=5):
    return obj.head(n=n)


@dispatch
def replace(col, old, new):
    raise NotImplementedError()


@replace.specialize("pandas", argument_type="Column")
def _replace_pandas(col, old, new):
    return col.replace(old, new, regex=False)


@replace.specialize("polars", argument_type="Column")
def _replace_polars(col, old, new):
    return col.replace(old, new)


def with_columns(df, **new_cols):
    cols = {col_name: col(df, col_name) for col_name in column_names(df)}
    cols.update({n: make_column_like(df, c, n) for n, c in new_cols.items()})
    return make_dataframe_like(df, cols)
