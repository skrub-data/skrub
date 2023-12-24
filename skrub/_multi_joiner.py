"""
The MultiJoiner and MultiAggJoiner extend Joiner and AggJoiner
to multiple auxiliary tables.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone

# from skrub._agg_joiner import AggJoiner
from skrub._joiner import DEFAULT_REF_DIST, DEFAULT_STRING_ENCODER  # Joiner


def check_multi_key():
    return


class MultiJoiner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        aux_table,
        *,
        key=None,
        main_key=None,
        aux_key=None,
        suffix="",
        max_dist=np.inf,
        ref_dist=DEFAULT_REF_DIST,
        string_encoder=DEFAULT_STRING_ENCODER,
        add_match_info=True,
    ):
        self.aux_table = aux_table
        self.main_key = main_key
        self.aux_key = aux_key
        self.key = key
        self.suffix = suffix
        self.max_dist = max_dist
        self.ref_dist = ref_dist
        self.string_encoder = (
            clone(string_encoder)
            if string_encoder is DEFAULT_STRING_ENCODER
            else string_encoder
        )
        self.add_match_info = add_match_info


class MultiAggJoiner(BaseEstimator, TransformerMixin):
    """Aggregate auxiliary dataframes before joining them on a base dataframe.

    Apply numerical and categorical aggregation operations on the columns
    to aggregate, selected by dtypes. See the list of supported operations
    at the parameter `operation`.

    The grouping columns used during the aggregation are the columns used
    as keys for joining.

    Accepts :obj:`pandas.DataFrame` and :class:`polars.DataFrame` inputs.

    Parameters
    ----------
    aux_tables : DataFrameLike or str or iterable
        Auxiliary dataframe to aggregate then join on the base table.
        The placeholder string "X" can be provided to perform
        self-aggregation on the input data.

    keys : str or list of str, default=None
        The column names to use for both ``main_key`` and ``aux_key`` when they
        are the same. Provide either ``key`` or both ``main_key`` and ``aux_key``.

    main_keys : str or iterable of str
        Select the columns from the main table to use as keys during
        the join operation.
        If main_key is a list, we will perform a multi-column join.

    aux_key : str, or iterable of str, or iterable of iterable of str
        Select the columns from the auxiliary dataframe to use as keys during
        the join operation.

    cols : str, or iterable of str, or iterable of iterable of str, default=None
        Select the columns from the auxiliary dataframe to use as values during
        the aggregation operations.
        If None, cols are all columns from table, except `aux_key`.

    operation : str or iterable of str, default=None
        Aggregation operations to perform on the auxiliary table.

        numerical : {"sum", "mean", "std", "min", "max", "hist", "value_counts"}
            'hist' and 'value_counts' accept an integer argument to parametrize
            the binning.

        categorical : {"mode", "count", "value_counts"}

        If set to None (the default), ['mean', 'mode'] will be used.

    suffix : str or iterable of str, default=None
        The suffixes that will be added to each table columns in case of
        duplicate column names.
        If set to None, the table index in 'aux_table' are used,
        e.g. for a duplicate columns: price (main table),
        price_1 (auxiliary table 1), price_2 (auxiliary table 2), etc.

    See Also
    --------
    AggJoiner :
        Aggregate an auxiliary dataframe before joining it on a base dataframe.
    """

    def __init__(
        self,
        aux_tables,
        *,
        keys=None,
        main_keys=None,
        aux_keys=None,
        cols=None,
        operation=None,
        suffix=None,
    ):
        self.aux_tables = aux_tables
        self.keys = keys
        self.main_keys = main_keys
        self.aux_keys = aux_keys
        self.cols = cols
        self.operation = operation
        self.suffix = suffix
