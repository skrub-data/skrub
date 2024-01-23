"""
Pandas specialization of the aggregate and join operation.
"""
import re
from itertools import product

import numpy as np
import pandas as pd

from skrub._utils import atleast_1d_or_none


def make_dataframe(X, index=None, dtypes=None):
    """Convert an dictionary of columns into a Pandas dataframe.

    Parameters
    ----------
    X : mapping from column name to 1d iterable
        Input data to convert.

    index : 1d array-like, default=None
        The index of the dataframe.

    dtypes : str, data type, Series or Mapping of column name -> data type, default=None
        Use a str, numpy.dtype, pandas.ExtensionDtype or Python type to
        cast entire pandas object to the same type. Alternatively, use a
        mapping, e.g. {col: dtype, ...}, where col is a column label and dtype is
        a numpy.dtype or Python type to cast one or more of the DataFrame's
        columns to column-specific types

    Returns
    -------
    X : Pandas dataframe
        Converted output.
    """
    df = pd.DataFrame(X, index=index)
    if dtypes is not None:
        # 'df.astype(None)' might raise a ValueError because
        # it tries to cast all columns to floats.
        df = df.astype(dtypes)
    return df


def make_series(X, index=None, name=None, dtype=None):
    """Convert an 1d array into a Pandas series.

    Parameters
    ----------
    X : 1d iterable
        Input data to convert.

    index : 1d array-like, default=None
        The index of the series.

    name : str, default=None
        The name of the series.

    dtype : str, numpy.dtype, or ExtensionDtype, default=None
        Data type for the output Series.

    Returns
    -------
    X : Pandas series
        Converted output.
    """
    series = pd.Series(X, index=index, name=name, dtype=dtype)
    return series


def aggregate(
    table,
    key,
    cols_to_agg,
    num_operations=("mean",),
    categ_operations=("mode",),
    suffix=None,
):
    """Aggregates a :obj:`pandas.DataFrame`.

    This function uses the ``dataframe.groupby(key).agg`` method from Pandas.

    Parameters
    ----------
    table : pd.DataFrame,
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

    suffix : str, optional
        The suffix appended to output columns.

    Returns
    -------
    group : pd.DataFrame,
        The aggregated output.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"'table' must be a pandas dataframe, got {type(table)!r}.")

    key = atleast_1d_or_none(key)
    cols_to_agg = atleast_1d_or_none(cols_to_agg)
    num_operations = atleast_1d_or_none(num_operations)
    categ_operations = atleast_1d_or_none(categ_operations)
    suffix = "" if suffix is None else suffix

    num_cols, categ_cols = split_num_categ_cols(table[cols_to_agg])

    num_named_agg, num_value_counts = get_named_agg(table, num_cols, num_operations)
    categ_named_agg, categ_value_counts = get_named_agg(
        table, categ_cols, categ_operations
    )

    named_agg = {**num_named_agg, **categ_named_agg}
    if named_agg:
        base_group = table.groupby(key).agg(**named_agg)
    else:
        base_group = None

    # 'histogram' and 'value_counts' requires a pivot
    value_counts = {**num_value_counts, **categ_value_counts}
    for output_key, (col_to_agg, kwargs) in value_counts.items():
        serie_group = table.groupby(key)[col_to_agg].value_counts(**kwargs)
        serie_group.name = output_key
        pivot = (
            serie_group.reset_index()
            .pivot(index=key, columns=col_to_agg)
            .reset_index()
            .fillna(0)
        )
        cols = pivot.columns.droplevel(0)
        index_cols = np.atleast_1d(key).tolist()
        feature_cols = (f"{col_to_agg}_" + cols[len(index_cols) :].astype(str)).tolist()
        cols = [*index_cols, *feature_cols]
        pivot.columns = cols

        if base_group is None:
            base_group = pivot
        else:
            base_group = base_group.merge(pivot, on=key, how="left")

    if base_group is None:
        raise ValueError("No aggregation to perform.")

    base_group.columns = [
        f"{col}{suffix}" if col not in key else col for col in base_group.columns
    ]
    sorted_cols = sorted(base_group.columns)

    return base_group[sorted_cols]


def join(
    left,
    right,
    left_on,
    right_on,
):
    """Left join two :obj:`pandas.DataFrame`.

    This function uses the ``dataframe.merge`` method from Pandas.

    Parameters
    ----------
    left : pd.DataFrame,
        The left dataframe to left-join.

    right : pd.DataFrame,
        The right dataframe to left-join.

    left_on : str or Iterable[str]
        Left keys to merge on.

    right_on : str or Iterable[str]
        Right keys to merge on.

    Returns
    -------
    merged : pd.DataFrame,
        The merged output.
    """
    if not (isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame)):
        raise TypeError(
            "'left' and 'right' must be pandas dataframes, "
            f"got {type(left)!r} and {type(right)!r}."
        )
    return left.merge(
        right,
        how="left",
        left_on=left_on,
        right_on=right_on,
    )


def get_named_agg(table, cols, operations):
    """Map aggregation tuples to their output key.

    The dictionary has the form: output_key = (column, aggfunc).
    This is used as input for the ``dataframe.agg`` method from Pandas.

    'value_counts' and 'hist' operation require to pivot
    the tables and treated in a separate mapping.

    Parameters
    ----------
    table : pd.DataFrame,
        Input dataframe, only used to compute bins values if
        'value_counts' or 'hist' are operations.

    cols : list,
        The columns to aggregate.

    operations : list,
        The reduce operations to perform.

    Returns
    -------
    named_agg : dict,
        Named aggregation mapping.

    value_counts : dict,
        ``value_counts`` operations mapping.
    """
    named_agg, value_counts = {}, {}
    for col, operation in product(cols, operations):
        op_root, bin_args = _parse_argument(operation)
        aggfunc, bin_args = _get_aggfunc(table[col], op_root, bin_args)

        output_key = f"{col}_{op_root}"
        # 'value_counts' change the index of the resulting frame
        # and must be treated separately.
        if aggfunc == "value_counts":
            value_counts[output_key] = (col, bin_args)
        else:
            named_agg[output_key] = (col, aggfunc)

    return named_agg, value_counts


def _parse_argument(operation):
    """Split a text input into a function name and its argument.

    Parameters
    ----------
    operation : str,
        The operation to parse.

    Returns
    -------
    operation_root : str,
        The name of the operation before parenthesis, if any.

    bin_args : int,
        The number of bin to create for ``hist`` or ``value_counts``.

    Examples
    --------
    >>> _parse_argument("hist(10)")
    ('hist', 10)
    """
    split = re.split("\\(.+\\)", operation)
    op_root = split[0]
    if len(split) > 1:
        # remove op_root
        bin_args = re.split(f"^{op_root}", operation)
        bin_args = bin_args[1]
        # remove parenthesis
        bin_args = re.sub("\\(|\\)", "", bin_args)
        bin_args = int(bin_args)
        return op_root, bin_args
    else:
        return op_root, None


PANDAS_OPS_MAPPING = {
    "mode": pd.Series.mode,
    "quantile": pd.Series.quantile,
    "hist": "value_counts",
}


def _get_aggfunc(serie, op_root, n_bins):
    """Map operation roots to their pandas agg functions.

    When args is provided for histogram or value_counts,
    we create args

    Parameters
    ----------
    serie : pd.Series,
        Input series, used to compute the bins if n_bins is provided.

    op_root : str,
        Operation root, the operation without the bin argument, if any.

    n_bins : int,
        The number of bin to create when value_counts or hist operation are used.

    Returns
    -------
    aggfunc : str or callable,
        The pandas agg functions to perform

    bins_args : dict,
        The bins to create when using value_counts or hist.
    """
    aggfunc = PANDAS_OPS_MAPPING.get(op_root, op_root)

    if n_bins is not None:
        # histogram and value_counts
        if aggfunc == "value_counts":
            # If bins is a number, we need to set a fix bin range,
            # otherwise bins edges will be defined dynamically for
            # each rows.
            min_, max_ = serie.min(), serie.max()
            bins = np.linspace(min_, max_, n_bins + 1)
            bins_args = dict(bins=bins)
        else:
            raise ValueError(
                f"Operator {op_root!r} doesn't take any argument, got {n_bins!r}"
            )
    else:
        bins_args = {}

    return aggfunc, bins_args


def split_num_categ_cols(table):
    """Split dataframe columns between numerical and categorical."""
    num_cols = table.select_dtypes("number").columns
    categ_cols = table.select_dtypes(["object", "string", "category"]).columns

    return num_cols, categ_cols


def select(dataframe, columns):
    return dataframe[columns]


def rename_columns(dataframe, renaming_function):
    return dataframe.rename(
        columns={c: renaming_function(c) for c in dataframe.columns}
    )
