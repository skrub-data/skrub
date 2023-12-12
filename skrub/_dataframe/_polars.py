"""
Polars specialization of the aggregate and join operations.
"""
try:
    import polars as pl
    import polars.selectors as cs

    POLARS_SETUP = True
except ImportError:
    POLARS_SETUP = False

from itertools import product

from skrub._utils import atleast_1d_or_none


def make_dataframe(X, index=None, dtypes=None):
    """Convert an dictionary of columns into a Polars dataframe.

    Parameters
    ----------
    X : mapping from column name to 1d iterable
        Input data to convert.

    index : 1d array-like, default=None
        Unused since polars doesn't use index.
        Only here for compatibility with Pandas.

    dtypes : DataType or mapping of column names to DataType, default=None
        Mapping of column names (or selector) to dtypes, or a single dtype
        to which all columns will be cast.

    Returns
    -------
    X : Polars dataframe
        Converted output.
    """
    if index is not None:
        raise ValueError(
            "Polars dataframes don't have an index, but "
            f"the Polars dataframe maker was called with {index=!r}."
        )
    df = pl.DataFrame(X)
    if dtypes is not None:
        df = df.cast(dtypes)
    return df


def make_series(X, index=None, name=None, dtype=None):
    """Convert an 1d array into a Polars series.

    Parameters
    ----------
    X : 1d iterable
        Input data to convert.

    index : 1d array-like, default=None
        Unused since polars doesn't use index.
        Only here for compatibility with Pandas.

    name : str, default=None
        The name of the series.

    dtype : DataType, default=None
        Polars dtype of the Series data.

    Returns
    -------
    X : Polars series
        Converted output.
    """
    if index is not None:
        raise ValueError(
            "Polars series don't have an index, but "
            f"the Polars series maker was called with {index=!r}."
        )
    series = pl.Series(values=X, name=name, dtype=dtype)
    return series


def aggregate(
    table,
    key,
    cols_to_agg,
    num_operations=("mean",),
    categ_operations=("mode",),
    suffix=None,
):
    """Aggregate a :obj:`polars.DataFrame` or :obj:`polars.LazyFrame`.

    This function uses the ``dataframe.group_by(key).agg`` method from Polars.

    Parameters
    ----------
    table : pl.DataFrame or pl.LazyFrame,
        The input dataframe to aggregate.

    key : str or Iterable[str],
        The columns used as keys to aggregate on.

    cols_to_agg : str or Iterable[str],
        The columns to aggregate.

    num_operations : str or Iterable[str],
        The reduction functions to apply on numerical columns
        in ``cols_to_agg`` during the aggregation.

    categ_operations : str or Iterable[str],
        The reduction functions to apply on categorical columns
        in ``cols_to_agg`` during the aggregation.

    suffix : str,
        The suffix appended to output columns.

    Returns
    -------
    group : pl.DataFrame or pl.LazyFrame,
        The aggregated output.
    """
    if not isinstance(table, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError(
            f"'table' must be a polars dataframe or lazyframe, got {type(table)!r}."
        )

    key = atleast_1d_or_none(key)
    cols_to_agg = atleast_1d_or_none(cols_to_agg)
    num_operations = atleast_1d_or_none(num_operations)
    categ_operations = atleast_1d_or_none(categ_operations)
    suffix = "" if suffix is None else suffix

    num_cols, categ_cols = split_num_categ_cols(table.select(cols_to_agg))

    num_aggfuncs, num_mode_cols = get_aggfuncs(num_cols, num_operations)
    categ_aggfuncs, categ_mode_cols = get_aggfuncs(categ_cols, categ_operations)

    aggfuncs = [*num_aggfuncs, *categ_aggfuncs]
    # If aggfuncs is empty, the output will be a series of index.
    table = table.group_by(key).agg(aggfuncs)

    # flattening post-processing of mode() cols
    flatten_ops = []
    for col in [*num_mode_cols, *categ_mode_cols]:
        flatten_ops.append(pl.col(col).list[0].alias(col))
    # add columns, no-op if 'flatten_ops' is empty.
    table = table.with_columns(flatten_ops)

    cols_renaming = {col: f"{col}{suffix}" for col in table.columns if col not in key}
    table = table.rename(cols_renaming)
    sorted_cols = sorted(table.columns)

    return table.select(sorted_cols)


def join(left, right, left_on, right_on):
    """Left join two :obj:`polars.DataFrame` or :obj:`polars.LazyFrame`.

    This function uses the ``dataframe.join`` method from Polars.

    Note that the input dataframes type must agree: either both
    Polars dataframes or both Polars lazyframes.

    Mixing polars dataframe with lazyframe will raise an error.

    Parameters
    ----------
    left : pl.DataFrame or pl.LazyFrame,
        The left dataframe of the left-join.

    right : pl.DataFrame or pl.LazyFrame,
        The right dataframe of the left-join.

    left_on : str or Iterable[str],
        Left keys to merge on.

    right_on : str or Iterable[str],
        Right keys to merge on.

    Returns
    -------
    merged : pl.DataFrame or pl.LazyFrame,
        The merged output.
    """
    is_dataframe = isinstance(left, pl.DataFrame) and isinstance(right, pl.DataFrame)
    is_lazyframe = isinstance(left, pl.LazyFrame) and isinstance(right, pl.LazyFrame)
    if is_dataframe or is_lazyframe:
        return left.join(
            right,
            how="left",
            left_on=left_on,
            right_on=right_on,
        )
    else:
        raise TypeError(
            "'left' and 'right' must be polars dataframes or lazyframes, "
            f"got {type(left)!r} and {type(right)!r}."
        )


def get_aggfuncs(cols, operations):
    """List Polars aggregation functions.

    The list is used as input for the ``dataframe.group_by().agg()`` method from Polars.
    The 'mode' operation needs a flattening post-processing.

    Parameters
    ----------
    cols : list,
        The columns to aggregate.

    operations : list,
        The reduce operations to perform.

    Returns
    -------
    aggfuncs : list,
        Named aggregation list.

    mode_cols : list,
        Output keys to post-process after 'mode' aggregation.
    """
    aggfuncs, mode_cols = [], []
    for col, operation in product(cols, operations):
        output_key = f"{col}_{operation}"
        aggfunc = _polars_ops_mapping(col, operation, output_key)
        aggfuncs.append(aggfunc)

        if operation == "mode":
            mode_cols.append(output_key)

    return aggfuncs, mode_cols


def _polars_ops_mapping(col, operation, output_key):
    """Map an operation to its Polars expression.

    Parameters
    ----------
    col : str,
        Name of the column to aggregate.
    operation : str,
        Name of the reduce function.
    output_key : str,
        Name of the reduced column.

    Returns
    -------
    aggfunc: polars.Expression,
        The expression to apply.
    """
    polars_aggfuncs = {
        "mean": pl.col(col).mean(),
        "std": pl.col(col).std(),
        "sum": pl.col(col).sum(),
        "min": pl.col(col).min(),
        "max": pl.col(col).max(),
        "mode": pl.col(col).mode(),
    }
    aggfunc = polars_aggfuncs.get(operation, None)

    if aggfunc is None:
        raise ValueError(
            f"Polars operation {operation!r} is not supported. Available:"
            f" {list(polars_aggfuncs)}"
        )

    return aggfunc.alias(output_key)


def split_num_categ_cols(table):
    """Split a dataframe columns between numerical and categorical."""
    num_cols = table.select(cs.numeric()).columns
    categ_cols = table.select(cs.string()).columns

    return num_cols, categ_cols


def select(dataframe, columns):
    return dataframe.select(columns)


def rename_columns(dataframe, renaming_function):
    return dataframe.rename({c: renaming_function(c) for c in dataframe.columns})
