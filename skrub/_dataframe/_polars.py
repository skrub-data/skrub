"""
Polars specialization of the aggregate and join operations.
"""
try:
    import polars as pl
    import polars.selectors as cs

except ImportError:
    pass

from itertools import product

from skrub._utils import atleast_1d_or_none

# TODO: _dataframe._polars is temporary; all code in this module should be moved
# elsewhere and use the dispatch mechanism.


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
    table : pl.DataFrame or pl.LazyFrame
        The input dataframe to aggregate.

    key : str or Iterable[str]
        The columns used as keys to aggregate on.

    cols_to_agg : str or Iterable[str]
        The columns to aggregate.

    num_operations : str or Iterable[str], default=("mean",)
        The reduction functions to apply on numerical columns
        in ``cols_to_agg`` during the aggregation.

    categ_operations : str or Iterable[str], default=("mode",)
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


def get_aggfuncs(cols, operations):
    """List Polars aggregation functions.

    The list is used as input for the ``dataframe.group_by().agg()`` method from Polars.
    The 'mode' operation needs a flattening post-processing.

    Parameters
    ----------
    cols : list
        The columns to aggregate.

    operations : list
        The reduce operations to perform.

    Returns
    -------
    aggfuncs : list
        Named aggregation list.

    mode_cols : list
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
    col : str
        Name of the column to aggregate.
    operation : str
        Name of the reduce function.
    output_key : str
        Name of the reduced column.

    Returns
    -------
    aggfunc: polars.Expression
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


def rename_columns(dataframe, renaming_function):
    return dataframe.rename({c: renaming_function(c) for c in dataframe.columns})
