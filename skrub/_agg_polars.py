try:
    import polars as pl
    import polars.selectors as cs

    # TODO: Enable polars accross the library
    POLARS_SETUP = True
except ImportError:
    POLARS_SETUP = False

from itertools import product


def aggregate(
    table,
    cols_to_join,
    cols_to_agg,
    num_ops,
    categ_ops,
    suffix,
):
    num_cols, categ_cols = split_num_categ_cols(table.select(cols_to_agg))

    num_ops, num_mode_cols = get_agg_ops(num_cols, num_ops)
    categ_ops, categ_mode_cols = get_agg_ops(categ_cols, categ_ops)

    all_ops = [*num_ops, *categ_ops]
    table = table.groupby(cols_to_join).agg(all_ops)

    # flattening post-processing of mode() cols
    flatten_ops = []
    for col in [*num_mode_cols, *categ_mode_cols]:
        flatten_ops.append(pl.col(col).list[0].alias(col))
    table = table.with_columns(flatten_ops)

    cols_renaming = {
        col: f"{col}{suffix}" for col in table.columns if col not in cols_to_join
    }
    table = table.rename(cols_renaming)
    sorted_cols = sorted(table.columns)

    return table[sorted_cols]


def join(left, right, left_on, right_on):
    return left.join(
        right,
        how="left",
        left_on=left_on,
        right_on=right_on,
    )


def split_num_categ_cols(table):
    num_cols = table.select(cs.numeric()).columns
    categ_cols = table.select(cs.string()).columns

    return num_cols, categ_cols


def get_agg_ops(cols, agg_ops):
    stats, mode_cols = [], []
    for col, op_name in product(cols, agg_ops):
        out_name = f"{col}_{op_name}"
        op_dict = {
            "mean": pl.col(col).mean().alias(out_name),
            "std": pl.col(col).std().alias(out_name),
            "sum": pl.col(col).sum().alias(out_name),
            "min": pl.col(col).min().alias(out_name),
            "max": pl.col(col).max().alias(out_name),
            "mode": pl.col(col).mode().alias(out_name),
        }
        op = op_dict.get(op_name, None)
        if op is None:
            raise ValueError(
                f"Polars operation '{op_name}' is not supported. Available:"
                f" {list(op_dict)}"
            )
        stats.append(op)

        # mode() output needs a flattening post-processing
        if op_name == "mode":
            mode_cols.append(out_name)

    return stats, mode_cols
