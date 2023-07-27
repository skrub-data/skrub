import re
from itertools import product

import numpy as np
import pandas as pd


def aggregate(
    table,
    cols_to_join,
    cols_to_agg,
    num_ops,
    categ_ops,
    suffix,
):
    num_cols, categ_cols = split_num_categ_cols(table[cols_to_agg])

    num_stats, num_reindexing = get_agg_ops(table, num_cols, num_ops)
    categ_stats, categ_reindexing = get_agg_ops(table, categ_cols, categ_ops)

    stats = {**num_stats, **categ_stats}
    if stats:
        base_group = table.groupby(cols_to_join).agg(**stats)
    else:
        base_group = None

    # 'histogram' and 'value_counts' requires a pivot
    reindexing = {**num_reindexing, **categ_reindexing}
    for key, (col_to_agg, kwargs) in reindexing.items():
        serie_group = table.groupby(cols_to_join)[col_to_agg].value_counts(**kwargs)
        serie_group.name = key
        pivot = (
            serie_group.reset_index()
            .pivot(index=cols_to_join, columns=col_to_agg)
            .reset_index()
            .fillna(0)
        )
        cols = pivot.columns.droplevel(0)
        index_cols = np.atleast_1d(cols_to_join).tolist()
        feature_cols = (f"{col_to_agg}_" + cols[len(index_cols) :].astype(str)).tolist()
        cols = [*index_cols, *feature_cols]
        pivot.columns = cols

        if base_group is None:
            base_group = pivot
        else:
            base_group = base_group.merge(pivot, on=cols_to_join, how="left")

    if base_group is None:
        raise ValueError("No aggregation has been performed")

    base_group.columns = [
        f"{col}{suffix}" if col not in cols_to_join else col
        for col in base_group.columns
    ]
    sorted_cols = sorted(base_group.columns)

    return base_group[sorted_cols]


def join(left, right, left_on, right_on):
    return left.merge(
        right,
        how="left",
        left_on=left_on,
        right_on=right_on,
    )


def get_agg_ops(table, cols, agg_ops):
    stats, value_counts_stats = {}, {}

    for col, op_name in product(cols, agg_ops):
        op_root, args = _parse_argument(op_name)
        op, args = _get_ops(table[col], op_root, args)

        key = f"{col}_{op_name}"
        # 'value_counts' change the index of the resulting frame
        # and must be treated separately.
        if op == "value_counts":
            value_counts_stats[key] = (col, args)
        else:
            stats[key] = (col, op)

    return stats, value_counts_stats


def _parse_argument(op_name):
    """Split a text input into a function name and its argument.

    Examples
    --------
    >>> _parse_argument("hist(10)")
    "hist", 10
    >>> _parse_argument("hist([2, 3])")
    "hist", [2, 3]
    """
    split = re.split(r"\(.+\)", op_name)
    op_root = split[0]
    if len(split) > 1:
        args = re.split(f"^{op_root}", op_name)
        args = args[1]
        args = re.sub(r"\(|\)", "", args)
        args = int(args)
        return op_root, args
    else:
        return op_root, None


PANDAS_OPS_MAPPING = {
    "mode": pd.Series.mode,
    "quantile": pd.Series.quantile,
    "hist": "value_counts",
}


def _get_ops(serie, op_root, args):
    op = PANDAS_OPS_MAPPING.get(op_root, op_root)

    if args is not None:
        # histogram and value_counts
        if op == "value_counts":
            # If bins is a number, we need to set a fix bin range,
            # otherwise bins edges will be defined dynamically for
            # each rows.
            min_, max_ = serie.min(), serie.max()
            args = np.linspace(min_, max_, args + 1)
            args = dict(bins=args)
        else:
            raise ValueError(
                f"Operator '{op_root}' doesn't take any argument, got '{args}'"
            )
    else:
        args = {}

    return op, args


def split_num_categ_cols(table):
    num_cols = table.select_dtypes("number").columns
    categ_cols = table.select_dtypes(["object", "string", "category"]).columns
    return num_cols, categ_cols
