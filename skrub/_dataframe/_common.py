import builtins
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pandas.api.types
from sklearn.utils.fixes import parse_version

from skrub import _join_utils

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
    "concat",
    "is_column_list",
    "to_column_list",
    "col",
    "col_by_idx",
    "collect",
    #
    # Querying, modifying metadata and shape
    #
    "shape",
    "to_frame",
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
    "is_duration",
    "is_categorical",
    "to_categorical",
    "is_all_null",
    #
    # Inspecting, selecting and modifying values
    #
    "all",
    "any",
    "sum",
    "min",
    "max",
    "std",
    "mean",
    "pearson_corr",
    "sort",
    "value_counts",
    "quantile",
    "is_null",
    "has_nulls",
    "drop_nulls",
    "fill_nulls",
    "n_unique",
    "unique",
    "filter",
    "where",
    "where_row",
    "sample",
    "head",
    "slice",
    "replace",
    "with_columns",
    "abs",
    "total_seconds",
    "is_sorted",
]

pandas_version = parse_version(parse_version(pd.__version__).base_version)

#
# Inspecting containers' type and module
# ======================================
#


def _raise(obj, kind="object"):
    raise TypeError(
        "Operation not supported on this object. Expecting a Pandas or Polars "
        f"{kind}, but got an object of type {type(obj)}."
    )


@dispatch
def dataframe_module_name(obj):
    """Return the dataframe module this object belongs to: 'pandas' or 'polars'."""
    raise _raise(obj)


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
    raise _raise(col, kind="Series")


@to_list.specialize("pandas", argument_type="Column")
def _to_list_pandas(col):
    result = col.tolist()
    return [None if item is pd.NA else item for item in result]


@to_list.specialize("polars", argument_type="Column")
def _to_list_polars(col):
    return col.to_list()


@dispatch
def to_numpy(col):
    raise _raise(col, kind="object")


@to_numpy.specialize("pandas", argument_type="Column")
def _to_numpy_pandas_column(col):
    if pd.api.types.is_numeric_dtype(col) and col.isna().any():
        col = col.astype(float)
    return col.to_numpy()


@to_numpy.specialize("polars", argument_type="Column")
def _to_numpy_polars_column(col):
    return col.to_numpy()


@to_numpy.specialize("pandas", argument_type="DataFrame")
def _to_numpy_pandas_table(df):
    return df.to_numpy()


@to_numpy.specialize("polars", argument_type="DataFrame")
def _to_numpy_polars_table(df):
    return df.to_numpy()


@dispatch
def to_pandas(obj):
    raise _raise(obj)


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
    raise _raise(obj)


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
    raise _raise(obj)


@make_column_like.specialize("pandas")
def _make_column_like_pandas(obj, values, name):
    return pd.Series(data=values, name=name)


@make_column_like.specialize("polars")
def _make_column_like_polars(obj, values, name):
    return pl.Series(values=values, name=name)


@dispatch
def null_value_for(obj):
    raise _raise(obj)


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
    raise _raise(col, kind="Series")


@all_null_like.specialize("pandas", argument_type="Column")
def _all_null_like_pandas(col, length=None, dtype=None, name=None):
    if length is None:
        length = len(col)
    if dtype is None:
        dtype = col.dtype
    if name is None:
        name = col.name
    return pd.Series(dtype=dtype, index=np.arange(length), name=name)


@all_null_like.specialize("polars", argument_type="Column")
def _all_null_like_polars(col, length=None, dtype=None, name=None):
    if length is None:
        length = len(col)
    if dtype is None:
        dtype = col.dtype
    if name is None:
        name = col.name
    return pl.Series(name, [None] * length, dtype=dtype)


def _check_same_type(objects):
    is_df = np.array([is_dataframe(obj) for obj in objects])
    is_col = np.array([is_column(obj) for obj in objects])
    is_other = ~(is_df | is_col)

    if is_df.all() or is_col.all() or is_other.all():
        return

    def _indices(array):
        return np.argwhere(array).ravel()

    prefix = "Mixing types is not allowed, got"
    hints = []
    if is_df.any():
        hints.append(f"dataframes at position {_indices(is_df)}")

    if is_col.any():
        hints.append(f"series at position {_indices(is_col)}")

    if is_other.any():
        hints.append(
            "types that are neither dataframes nor series at position "
            f"{_indices(is_other)}"
        )

    msg = prefix + " " + ", ".join(hints) + "."

    raise TypeError(msg)


@dispatch
def concat(*dataframes, axis=0):
    # This is accessed only when the first element of *dataframes is neither
    # a pandas or polars valid type.
    raise _raise(dataframes[0])


@concat.specialize("pandas", argument_type="DataFrame")
def _concat_pandas(*dataframes, axis=0):
    _check_same_type(dataframes)
    kwargs = {"copy": False} if pandas_version < parse_version("3.0") else {}
    if axis == 0:
        return pd.concat(dataframes, axis=0, ignore_index=True, **kwargs)
    else:  # axis == 1
        init_index = dataframes[0].index
        dataframes = [df.reset_index(drop=True) for df in dataframes]
        dataframes = _join_utils.make_column_names_unique(*dataframes)
        result = pd.concat(dataframes, axis=1, **kwargs)
        result.index = init_index
        return result


@concat.specialize("polars", argument_type="DataFrame")
def _concat_polars(*dataframes, axis=0):
    _check_same_type(dataframes)
    if axis == 0:
        return pl.concat(dataframes, how="diagonal_relaxed")
    else:  # axis == 1
        dataframes = _join_utils.make_column_names_unique(*dataframes)
        return pl.concat(dataframes, how="horizontal")


@concat.specialize("pandas", argument_type="Column")
def _concat_pandas_cols(*columns, axis=0):
    _check_same_type(columns)
    return _concat_pandas(*[to_frame(col) for col in columns], axis=axis)


@concat.specialize("polars", argument_type="Column")
def _concat_polars_cols(*columns, axis=0):
    _check_same_type(columns)
    return _concat_polars(*[to_frame(col) for col in columns], axis=axis)


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
        return [col_by_idx(obj, idx) for idx in range(shape(obj)[1])]
    if not is_column_list(obj):
        raise TypeError("obj should be a DataFrame, a Column or a list of Columns.")
    return obj


@dispatch
def col(df, col_name):
    raise _raise(df, kind="DataFrame")


@col.specialize("pandas", argument_type="DataFrame")
def _col_pandas(df, col_name):
    return df[col_name]


@col.specialize("polars", argument_type="DataFrame")
def _col_polars(df, col_name):
    return df[col_name]


@dispatch
def col_by_idx(df, col_idx):
    raise _raise(df, kind="DataFrame")


@col_by_idx.specialize("pandas", argument_type="DataFrame")
def _col_by_idx_pandas(df, col_idx):
    return df.iloc[:, col_idx]


@col_by_idx.specialize("polars", argument_type="DataFrame")
def _col_by_idx_polars(df, col_idx):
    return df[df.columns[col_idx]]


@dispatch
def collect(df):
    return df


@collect.specialize("polars", argument_type="LazyFrame")
def _collect_polars_lazyframe(df):
    return df.collect()


#
# Querying, modifying metadata and shape
# ======================================
#


@dispatch
def shape(obj):
    raise _raise(obj)


@shape.specialize("pandas")
def _shape_pandas(obj):
    return obj.shape


@shape.specialize("polars")
def _shape_polars(obj):
    return obj.shape


@dispatch
def to_frame(col):
    """Convert a single Column to a DataFrame."""
    raise _raise(col, kind="Series")


@to_frame.specialize("pandas", argument_type="Column")
def _to_frame_pandas(col):
    return col.to_frame()


@to_frame.specialize("polars", argument_type="Column")
def _to_frame_polars(col):
    return col.to_frame()


@dispatch
def name(col):
    raise _raise(col, kind="Series")


@name.specialize("pandas", argument_type="Column")
def _name_pandas(col):
    return col.name


@name.specialize("polars", argument_type="Column")
def _name_polars(col):
    return col.name


@dispatch
def column_names(df):
    raise _raise(df, kind="DataFrame")


@column_names.specialize("pandas", argument_type="DataFrame")
def _column_names_pandas(df):
    return list(df.columns.values)


@column_names.specialize("polars", argument_type="DataFrame")
def _column_names_polars(df):
    return df.columns


@dispatch
def rename(col, new_name):
    raise _raise(col, kind="Series")


@rename.specialize("pandas", argument_type="Column")
def _rename_pandas(col, new_name):
    return col.rename(new_name)


@rename.specialize("polars", argument_type="Column")
def _rename_polars(col, new_name):
    return col.rename(new_name)


@dispatch
def set_column_names(df, new_col_names):
    raise _raise(df, kind="DataFrame")


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
    raise _raise(col, kind="Series")


@dtype.specialize("pandas", argument_type="Column")
def _dtype_pandas(col):
    return col.dtype


@dtype.specialize("polars", argument_type="Column")
def _dtype_polars(col):
    return col.dtype


@dispatch
def dtypes(df):
    raise _raise(df, kind="DataFrame")


@dtypes.specialize("pandas", argument_type="DataFrame")
def _dtypes_pandas(df):
    return list(df.dtypes.values)


@dtypes.specialize("polars", argument_type="DataFrame")
def _dtypes_polars(df):
    return df.dtypes


@dispatch
def cast(col, dtype):
    raise _raise(col, kind="Series")


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
    raise _raise(obj)


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
    raise _raise(col, kind="Series")


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
    raise _raise(col, kind="Series")


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
    raise _raise(col, kind="Series")


@is_integer.specialize("pandas", argument_type="Column")
def _is_integer_pandas(col):
    return pd.api.types.is_integer_dtype(col)


@is_integer.specialize("polars", argument_type="Column")
def _is_integer_polars(col):
    return col.dtype.is_integer()


@dispatch
def is_float(col):
    raise _raise(col, kind="Series")


@is_float.specialize("pandas", argument_type="Column")
def _is_float_pandas(col):
    return pd.api.types.is_float_dtype(col)


@is_float.specialize("polars", argument_type="Column")
def _is_float_polars(col):
    return col.dtype.is_float()


@dispatch
def to_float32(col, strict=True):
    raise _raise(col, kind="Series")


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
    raise _raise(col, kind="Series")


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
    raise _raise(col, kind="Series")


@to_string.specialize("pandas", argument_type="Column")
def _to_string_pandas(col):
    is_na = col.isna()
    if not (is_pandas_object(col) and is_string(col)):
        col = col.astype("str")
        col[is_na] = np.nan
        return col
    if not is_na.any():
        return col
    return _fill_nulls_pandas(col, np.nan)


@to_string.specialize("polars", argument_type="Column")
def _to_string_polars(col):
    # polars raises an error when trying to cast those types to string directly
    # so we have to use map_elements
    if col.dtype not in (pl.Object, pl.List, pl.Array):
        return _cast_polars(col, pl.String)
    # polars emits a performance warning when using map_elements
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=pl.exceptions.PolarsInefficientMapWarning
        )
        return col.map_elements(str, return_dtype=pl.String)


@dispatch
def is_object(col):
    raise _raise(col, kind="Series")


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
    raise _raise(col, kind="Series")


@is_any_date.specialize("pandas", argument_type="Column")
def _is_any_date_pandas(col):
    return pandas.api.types.is_datetime64_any_dtype(col)


@is_any_date.specialize("polars", argument_type="Column")
def _is_any_date_polars(col):
    return col.dtype in (pl.Date, pl.Datetime)


@dispatch
def to_datetime(col, format, strict=True):
    raise _raise(col, kind="Series")


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
    except (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError) as e:
        raise ValueError("Failed to convert to datetime") from e


@dispatch
def is_duration(col):
    raise _raise(col, kind="Series")


@is_duration.specialize("pandas", argument_type="Column")
def _is_duration_pandas(col):
    return pd.api.types.is_timedelta64_dtype(col)


@is_duration.specialize("polars", argument_type="Column")
def _is_duration_polars(col):
    return col.dtype == pl.Duration


@dispatch
def is_categorical(col):
    raise _raise(col, kind="Series")


@is_categorical.specialize("pandas", argument_type="Column")
def _is_categorical_pandas(col):
    return isinstance(col.dtype, pd.CategoricalDtype)


@is_categorical.specialize("polars", argument_type="Column")
def _is_categorical_polars(col):
    return col.dtype in (pl.Categorical, pl.Enum)


@dispatch
def to_categorical(col):
    raise _raise(col, kind="Series")


@to_categorical.specialize("pandas", argument_type="Column")
def _to_categorical_pandas(col):
    return _cast_pandas(col, "category")


@to_categorical.specialize("polars", argument_type="Column")
def _to_categorical_polars(col):
    if col.dtype in (pl.Categorical, pl.Enum):
        return col
    col = to_string(col)
    return _cast_polars(col, pl.Categorical())


@dispatch
def is_all_null(col):
    raise _raise(col, kind="Series")


@is_all_null.specialize("pandas", argument_type="Column")
def _is_all_null_pandas(col):
    return all(is_null(col))


@is_all_null.specialize("polars", argument_type="Column")
def _is_all_null_polars(col):
    # Column type is Null
    if col.dtype == pl.Null:
        return True
    # Column type is not Null, but all values are nulls: more efficient
    if col.null_count() == col.len():
        return True
    # Column type is not Null, not all values are null (check if NaN etc.): slower
    return all(is_null(col))


#
# Inspecting, selecting and modifying values
# ==========================================
#


@dispatch
def all(col):
    raise _raise(col, kind="Series")


@all.specialize("pandas", argument_type="Column")
def _all_pandas(col):
    return col.all()


@all.specialize("polars", argument_type="Column")
def _all_polars(col):
    return col.all()


@dispatch
def any(col):
    raise _raise(col, kind="Series")


@any.specialize("pandas", argument_type="Column")
def _any_pandas(col):
    return col.any()


@any.specialize("polars", argument_type="Column")
def _any_polars(col):
    return col.any()


@dispatch
def sum(col):
    raise _raise(col, kind="Series")


@sum.specialize("pandas", argument_type="Column")
def _sum_pandas_col(col):
    return col.sum()


@sum.specialize("polars", argument_type="Column")
def _sum_polars_col(col):
    return col.sum()


@dispatch
def min(col):
    raise _raise(col, kind="Series")


@min.specialize("pandas", argument_type="Column")
def _min_pandas_col(col):
    return col.min()


@min.specialize("polars", argument_type="Column")
def _min_polars_col(col):
    return col.min()


@dispatch
def max(col):
    raise _raise(col, kind="Series")


@max.specialize("pandas", argument_type="Column")
def _max_pandas_col(col):
    return col.max()


@max.specialize("polars", argument_type="Column")
def _max_polars_col(col):
    return col.max()


@dispatch
def std(col):
    raise _raise(col, kind="Series")


@std.specialize("pandas", argument_type="Column")
def _std_pandas_col(col):
    return col.std()


@std.specialize("polars", argument_type="Column")
def _std_polars_col(col):
    return col.std()


@dispatch
def mean(col):
    raise _raise(col, kind="Series")


@mean.specialize("pandas", argument_type="Column")
def _mean_pandas_col(col):
    return col.mean()


@mean.specialize("polars", argument_type="Column")
def _mean_polars_col(col):
    return col.mean()


@dispatch
def pearson_corr(df):
    raise _raise(df, kind="DataFrame")


@pearson_corr.specialize("pandas", argument_type="DataFrame")
def _pearson_corr_pandas(df):
    return df.corr(method="pearson", numeric_only=True)


@pearson_corr.specialize("polars", argument_type="DataFrame")
def _pearson_corr_polars(df):
    return pl.from_pandas(_pearson_corr_pandas(df.to_pandas()))


@dispatch
def value_counts(col):
    raise _raise(col, kind="Series")


@value_counts.specialize("pandas", argument_type="Column")
def _value_counts_pandas(col):
    return (
        col.value_counts(dropna=True)
        .reset_index()
        .set_axis(["value", "count"], axis="columns")
    )


@value_counts.specialize("polars", argument_type="Column")
def _value_counts_polars(col):
    return (
        col.drop_nulls()
        .rename("value")
        .value_counts()
        .with_columns(pl.col("count").cast(pl.Int64))
    )


@dispatch
def sort(df, by, descending=False):
    raise _raise(df, kind="DataFrame")


@sort.specialize("pandas", argument_type="DataFrame")
def _sort_pandas_dataframe(df, by, descending=False):
    return df.sort_values(
        by=by, ascending=not descending, ignore_index=True, na_position="last"
    )


@sort.specialize("polars", argument_type="DataFrame")
def _sort_polars_dataframe(df, by, descending=False):
    return df.sort(by=by, descending=descending, nulls_last=True)


@dispatch
def quantile(col, q, interpolation="nearest"):
    raise _raise(col, kind="Series")


@quantile.specialize("pandas", argument_type="Column")
def _quantile_pandas_column(col, q, interpolation="nearest"):
    return col.quantile(q, interpolation=interpolation)


@quantile.specialize("polars", argument_type="Column")
def _quantile_polars_column(col, q, interpolation="nearest"):
    return _drop_nulls_polars(col).quantile(q, interpolation=interpolation)


@dispatch
def is_null(col):
    raise _raise(col, kind="Series")


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
    raise _raise(col, kind="Series")


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
    raise _raise(obj)


@fill_nulls.specialize("pandas")
def _fill_nulls_pandas(obj, value):
    if parse_version(pd.__version__) < parse_version("2.2.0"):
        return obj.fillna(value)
    with pd.option_context("future.no_silent_downcasting", True):
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
    raise _raise(col, kind="Series")


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
    raise _raise(col, kind="Series")


@unique.specialize("pandas", argument_type="Column")
def _unique_pandas(col):
    return pd.Series(col.dropna().unique()).rename(col.name)


@unique.specialize("polars", argument_type="Column")
def _unique_polars(col):
    return col.unique().drop_nulls()


@dispatch
def filter(obj, predicate):
    raise _raise(obj)


@filter.specialize("pandas")
def _filter_pandas(obj, predicate):
    return obj[predicate]


@filter.specialize("polars")
def _filter_polars(obj, predicate):
    return obj.filter(predicate)


@dispatch
def where(col, mask, other):
    raise _raise(col, kind="Series")


@where.specialize("pandas", argument_type="Column")
def _where_pandas(col, mask, other):
    return col.where(mask, pd.Series(other))


@where.specialize("polars", argument_type="Column")
def _where_polars(col, mask, other):
    return col.zip_with(mask, pl.Series(other))


@dispatch
def where_row(obj, mask, other):
    raise _raise(obj)


@where_row.specialize("pandas")
def _where_row_pandas(obj, mask, other):
    return obj.apply(pd.Series.where, cond=mask, other=other)


@where_row.specialize("polars")
def _where_row_polars(obj, mask, other):
    return obj.with_columns(
        pl.when(pl.Series(mask)).then(pl.all()).otherwise(pl.Series(other))
    )


@dispatch
def sample(obj, n, seed=None):
    raise _raise(obj)


@sample.specialize("pandas")
def _sample_pandas(obj, n, seed=None):
    return obj.sample(n=n, random_state=seed, replace=False)


@sample.specialize("polars")
def _sample_polars(obj, n, seed=None):
    return obj.sample(n=n, seed=seed, with_replacement=False)


@dispatch
def head(obj, n=5):
    raise _raise(obj)


@head.specialize("pandas")
def _head_pandas(obj, n=5):
    return obj.head(n=n)


@head.specialize("polars")
def _head_polars(obj, n=5):
    return obj.head(n=n)


@dispatch
def slice(obj, *start_stop):
    raise _raise(obj)


@slice.specialize("pandas")
def _slice_pandas(obj, *start_stop):
    return obj.iloc[builtins.slice(*start_stop)]


@slice.specialize("polars")
def _slice_polars(obj, *start_stop):
    start, stop, _ = builtins.slice(*start_stop).indices(shape(obj)[0])
    return obj.slice(start, stop - start)


@dispatch
def replace(col, old, new):
    raise _raise(col, kind="Series")


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


@dispatch
def abs(col):
    raise _raise(col, kind="Series")


@abs.specialize("pandas", argument_type="Column")
def _abs_pandas(col):
    return col.abs()


@abs.specialize("polars", argument_type="Column")
def _abs_polars(col):
    return col.abs()


@dispatch
def total_seconds(col):
    raise _raise(col, kind="Series")


@total_seconds.specialize("pandas", argument_type="Column")
def _total_seconds_pandas(col):
    return col.dt.total_seconds()


@total_seconds.specialize("polars", argument_type="Column")
def _total_seconds_polars(col):
    return col.dt.total_microseconds().cast(float) * 1e-6


@dispatch
def is_sorted(col):
    """Check if a column is sorted."""
    raise _raise(col, kind="Series")


@is_sorted.specialize("pandas", argument_type="Column")
def _is_sorted_pandas(col):
    return col.is_monotonic_increasing or col.is_monotonic_decreasing


@is_sorted.specialize("polars", argument_type="Column")
def _is_sorted_polars(col):
    return col.is_sorted()
