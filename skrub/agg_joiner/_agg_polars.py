"""
Polars specialization of the aggregate and join operation
performed by AggJoiner and AggTarget.
"""
from skrub._utils import POLARS_SETUP, DataFrameLike

if POLARS_SETUP:
    import polars as pl
    import polars.selectors as cs

from itertools import product


def aggregate(
    table: DataFrameLike,
    cols_to_join: list[str],
    cols_to_agg: list[str],
    num_operations: list[str] | tuple[str] = ("mean",),
    categ_operations: list[str] | tuple[str] = ("mode",),
    suffix: str = "",
) -> DataFrameLike:
    """Aggregate the auxiliary tables.

    This function is only called during ``fit`` by AggJoiner and AggTarget.
    It uses the ``dataframe.groupby(key).agg`` method from Polars.

    Parameters
    ----------
    table : pl.DataFrame,
        The input auxiliary table to aggregate.

    cols_to_join : list,
        The columns used as keys to aggregate on.
        We also use the columns as keys during the subsequent join operation.

    cols_to_agg : list,
        The columns to aggregate.

    num_operations : list,
        The reduction functions to apply on numerical columns
        in ``cols_to_agg`` during the aggregation.

    categ_operations : list,
        The reduction functions to apply on categorical columns
        in ``cols_to_agg`` during the aggregation.

    suffix : str,
        The suffix appended to output columns to identify the input auxiliary table.

    Returns
    -------
    group : pl.DataFrame,
        The aggregated output.
    """
    num_cols, categ_cols = split_num_categ_cols(table.select(cols_to_agg))

    num_aggfuncs, num_mode_cols = get_aggfuncs(num_cols, num_operations)
    categ_aggfuncs, categ_mode_cols = get_aggfuncs(categ_cols, categ_operations)

    aggfuncs = [*num_aggfuncs, *categ_aggfuncs]
    table = table.groupby(cols_to_join).agg(aggfuncs)

    # flattening post-processing of mode() cols
    flatten_ops = []
    for col in [*num_mode_cols, *categ_mode_cols]:
        flatten_ops.append(pl.col(col).list[0].alias(col))
    # add columns, no-op if 'flatten_ops' is empty.
    table = table.with_columns(flatten_ops)

    cols_renaming = {
        col: f"{col}{suffix}" for col in table.columns if col not in cols_to_join
    }
    table = table.rename(cols_renaming)
    sorted_cols = sorted(table.columns)

    return table.select(sorted_cols)


def join(
    left: DataFrameLike,
    right: DataFrameLike,
    left_on: str | list[str],
    right_on: str | list[str],
) -> DataFrameLike:
    """Left join the main table to one aggregated auxilary table.

    This function is called during ``transform`` by the AggJoiner and AggTarget.
    It only uses the ``dataframe.join`` method from Polars.

    Parameters
    ----------
    left : pl.DataFrame,
        The main table.

    right : pl.DataFrame,
        The aggregated auxiliary table.

    left_on : str or array-like,
        Left keys to merge on.

    right_on : str or array-like,
        Right keys to merge on.

    Returns
    -------
    merged : pl.DataFrame,
        The merged output.
    """
    return left.join(
        right,
        how="left",
        left_on=left_on,
        right_on=right_on,
    )


def get_aggfuncs(
    cols: list[str],
    operations: list[str],
) -> tuple[list, list]:
    """List Polars aggregation functions.

    The list is used as input for the ``dataframe.agg`` method from Polars.
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
    """Split dataframe columns between numerical and categorical."""
    num_cols = table.select(cs.numeric()).columns
    categ_cols = table.select(cs.string()).columns

    return num_cols, categ_cols
