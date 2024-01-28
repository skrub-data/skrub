try:
    import pandas as pd
    import pandas.api.types
except ImportError:
    pass
try:
    import polars as pl
except ImportError:
    pass

from ._dispatch import dispatch

__all__ = [
    "skrub_namespace",
    "dataframe_module_name",
    "is_pandas",
    "is_polars",
    "shape",
    "is_dataframe",
    "is_lazyframe",
    "is_column",
    "col",
    "to_array",
    "pandas_convert_dtypes",
    "column_names",
    "name",
    "column_like",
    "dataframe_from_columns",
    "to_column_list",
    "is_in",
    "is_null",
    "drop_nulls",
    "n_unique",
    "is_bool",
    "is_numeric",
    "to_numeric",
    "is_string",
    "is_object",
    "is_anydate",
    "set_column_names",
    "collect",
    "is_categorical",
    "to_categorical",
    "to_datetime",
    "unique",
    "dtype",
    "cast",
    "where",
    "sample",
    "replace",
    "replace_regex",
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


def is_pandas(obj):
    return dataframe_module_name(obj) == "pandas"


def is_polars(obj):
    return dataframe_module_name(obj) == "polars"


@dispatch
def is_dataframe(obj):
    return False


@is_dataframe.specialize("pandas", "DataFrame")
def _is_dataframe_pandas(obj):
    return True


@is_dataframe.specialize("polars", ["DataFrame", "LazyFrame"])
def _is_dataframe_polars(obj):
    return True


@dispatch
def is_column(obj):
    return False


@is_column.specialize("pandas", "Column")
def _is_column_pandas(obj):
    return True


@is_column.specialize("polars", "Column")
def _is_column_polars(obj):
    return True


@dispatch
def to_array(obj):
    raise NotImplementedError()


@to_array.specialize("pandas")
def _to_array_pandas(obj):
    return obj.to_numpy()


@to_array.specialize("polars")
def _to_array_polars(obj):
    return obj.to_numpy()


@dispatch
def pandas_convert_dtypes(obj):
    return obj


@pandas_convert_dtypes.specialize("pandas")
def _pandas_convert_dtypes_pandas(obj):
    return obj.convert_dtypes()


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
def column_names(df):
    raise NotImplementedError()


@column_names.specialize("pandas")
def _column_names_pandas(df):
    return list(df.columns.values)


@column_names.specialize("polars")
def _column_names_polars(df):
    return df.columns


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
def column_like(column, values, name):
    return NotImplementedError()


@column_like.specialize("pandas")
def _column_like_pandas(column, values, name):
    return pd.Series(values, name=name)


@column_like.specialize("polars")
def _column_like_polars(column, values, name):
    return pl.Series(values, name=name)


@dispatch
def dataframe_from_columns(*columns):
    return NotImplementedError()


@dataframe_from_columns.specialize("pandas")
def _dataframe_from_columns_pandas(*columns):
    return pd.DataFrame({name(c): c for c in columns})


@dataframe_from_columns.specialize("polars")
def _dataframe_from_columns_polars(*columns):
    return pl.DataFrame({name(c): c for c in columns})


def to_column_list(obj):
    if is_column(obj):
        return [obj]
    if is_dataframe(obj):
        return [col(obj, c) for c in column_names(obj)]
    if not hasattr(obj, "__iter__"):
        raise TypeError("obj should be a DataFrame, a Column or a list of Columns.")
    return obj


@dispatch
def is_in(column, values):
    raise NotImplementedError()


@is_in.specialize("pandas")
def _is_in_pandas(column, values):
    return column.isin(values)


@is_in.specialize("polars")
def _is_in_polars(column, values):
    return column.is_in(values)


@dispatch
def is_null(column):
    raise NotImplementedError()


@is_null.specialize("pandas")
def _is_null_pandas(column):
    return column.isna()


@is_null.specialize("polars")
def _is_null_polars(column):
    return column.is_null()


@dispatch
def drop_nulls(column):
    raise NotImplementedError()


@drop_nulls.specialize("pandas")
def _drop_nulls_pandas(column):
    return column.dropna()


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
    return column.n_unique()


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
    return pandas.api.types.is_numeric_dtype(column)


@is_numeric.specialize("polars")
def _is_numeric_polars(column):
    return column.dtype.is_numeric()


@dispatch
def to_numeric(column, dtype=None):
    raise NotImplementedError()


@to_numeric.specialize("pandas")
def _to_numeric_pandas(column, dtype=None):
    return pd.to_numeric(column).astype(dtype)


@to_numeric.specialize("polars")
def _to_numeric_polars(column, dtype=None):
    if dtype is not None:
        return column.cast(dtype)
    if column.dtype.is_numeric():
        return column
    error = None
    for dtype in [pl.Int32, pl.Int64, pl.Float64]:
        try:
            return column.cast(dtype)
        except Exception as e:
            error = e
    raise ValueError("Could not convert column to numeric dtype") from error


@dispatch
def is_string(column):
    raise NotImplementedError()


@is_string.specialize("pandas")
def _is_string_pandas(column):
    return pandas.api.types.is_string_dtype(column)


@is_string.specialize("polars")
def _is_string_polars(column):
    return column.dtype == pl.String


@dispatch
def is_object(column):
    raise NotImplementedError()


@is_object.specialize("pandas")
def _is_object_pandas(column):
    if pandas.api.types.is_string_dtype(column):
        return False
    return pandas.api.types.is_object_dtype(column)


@is_object.specialize("polars")
def _is_object_polars(column):
    return column.dtype == pl.Object


@dispatch
def is_anydate(column):
    raise NotImplementedError()


@is_anydate.specialize("pandas")
def _is_anydate_pandas(column):
    return pandas.api.types.is_datetime64_any_dtype(column)


@is_anydate.specialize("polars")
def _is_anydate_polars(column):
    return column.dtype in (pl.Date, pl.Datetime)


@dispatch
def is_lazyframe(df):
    return False


@is_lazyframe.specialize("polars", "LazyFrame")
def _is_lazyframe_polars_lazyframe(df):
    return True


@dispatch
def collect(df):
    return df


@collect.specialize("polars", "LazyFrame")
def _collect_polars_lazyframe(df):
    return df.collect()


@dispatch
def set_column_names(df, new_column_names):
    raise NotImplementedError()


@set_column_names.specialize("pandas")
def _set_column_names_pandas(df, new_column_names):
    return df.set_axis(new_column_names, axis=1)


@set_column_names.specialize("polars")
def _set_column_names_polars(df, new_column_names):
    return df.rename(dict(zip(df.columns, new_column_names)))


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


@dispatch
def to_datetime(column, format):
    raise NotImplementedError()


@to_datetime.specialize("pandas")
def _to_datetime_pandas(column, format):
    if _is_anydate_pandas(column):
        return column
    return pd.to_datetime(column, format=format)


@to_datetime.specialize("polars")
def _to_datetime_polars(column, format):
    if _is_anydate_polars(column):
        return column
    return column.str.to_datetime(format=format)


@dispatch
def unique(column):
    raise NotImplementedError()


@unique.specialize("pandas")
def _unique_pandas(column):
    return pd.Series(column.dropna().unique())


@unique.specialize("polars")
def _unique_polars(column):
    return column.unique().drop_nulls()


@dispatch
def dtype(column):
    return column.dtype


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
    raise NotImplementedError()


@replace_regex.specialize("pandas")
def _replace_regex_pandas(column, pattern, replacement):
    return column.str.replace(pattern, replacement, regex=True)


@replace_regex.specialize("polars")
def _replace_regex_polars(column, pattern, replacement):
    return column.str.replace_all(pattern, replacement, literal=False)
