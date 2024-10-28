"""
Implement AggJoiner and AggTarget to join a main table to an auxiliary table,
with one-to-many relationships.

Both classes aggregate the auxiliary table first, then join this grouped
table with the main table.
"""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from skrub import _dataframe as sbd
from skrub import _join_utils
from skrub import _selectors as s
from skrub._dispatch import dispatch
from skrub._utils import atleast_1d_or_none

from ._check_input import CheckInputDataFrame

try:
    import polars as pl
except ImportError:
    pass

SUPPORTED_OPS = ["count", "mode", "min", "max", "sum", "median", "mean", "std"]
# summing strings works in pandas, not in polars
NUM_ONLY_OPS = ["sum", "median", "mean", "std"]


def aggregate(table, key, cols_to_agg, operations, suffix):
    """Aggregate the `table` by `key` and compute statistics for `cols_to_agg`.

    Parameters
    ----------
    table : DataFrame
        The input dataframe to aggregate.

    key : iterable of str
        The columns used as keys to aggregate on.

    cols_to_agg : iterable of str
        The columns to aggregate.

    operations : iterable of str
        The reduction functions to apply on columns in ``cols_to_agg``
        during the aggregation.

        Supported operations are "count", "mode", "min", "max", "sum", "median",
        "mean", "std". The operations "sum", "median", "mean", "std" are reserved
        to numeric type columns.

    suffix : str
        The suffix appended to output columns. Will only be applied
        to columns created by the aggregations.

    Returns
    -------
    DataFrame
        The aggregated output.
    """
    table_to_agg = s.select(table, s.make_selector(key) | s.make_selector(cols_to_agg))

    # Don't check the ID column, as it's not the one we aggregate on
    table_to_check = s.select(table_to_agg, ~s.cols(*key))
    not_numeric_cols = (~s.numeric()).expand(table_to_check)

    num_only_op = list(set(operations).intersection(set(NUM_ONLY_OPS)))

    if (len(not_numeric_cols) > 0) & (len(num_only_op) > 0):
        raise AttributeError(
            f"The operations {NUM_ONLY_OPS} are restricted to numeric columns."
            f" \nConsider removing the following columns: {not_numeric_cols} or the"
            f" following operations: {num_only_op}."
        )

    aggregated = perform_groupby(table_to_agg, key, cols_to_agg, operations)

    new_col_names = [
        f"{col}{suffix}" if col not in key else col
        for col in sbd.column_names(aggregated)
    ]
    new_col_names = _join_utils.pick_column_names(new_col_names)
    aggregated = sbd.set_column_names(aggregated, new_col_names)

    return aggregated


@dispatch
def perform_groupby(table, key, cols_to_agg, operations):
    raise NotImplementedError()


@perform_groupby.specialize("pandas", argument_type="DataFrame")
def _perform_groupby_pandas(table, key, cols_to_agg, operations):
    pandas_aggfuncs = {
        "mode": pd.Series.mode,
    }
    named_agg = {}
    for col, operation in product(cols_to_agg, operations):
        aggfunc = pandas_aggfuncs.get(operation, operation)
        output_key = f"{col}_{operation}"
        named_agg[output_key] = (col, aggfunc)

    aggregated = table.groupby(key).agg(**named_agg).reset_index(drop=False)

    return aggregated


@perform_groupby.specialize("polars", argument_type="DataFrame")
def _perform_groupby_polars(table, key, cols_to_agg, operations):
    aggfuncs = []
    for col, operation in product(cols_to_agg, operations):
        polars_aggfuncs = {
            "count": pl.col(col).count(),
            "median": pl.col(col).median(),
            "mean": pl.col(col).mean(),
            "std": pl.col(col).std(),
            "sum": pl.col(col).sum(),
            "min": pl.col(col).min(),
            "max": pl.col(col).max(),
            "mode": pl.col(col).mode().first(),
        }
        output_key = f"{col}_{operation}"
        aggfunc = polars_aggfuncs[operation].alias(output_key)
        aggfuncs.append(aggfunc)

    aggregated = table.group_by(key).agg(aggfuncs)

    return aggregated


class AggJoiner(TransformerMixin, BaseEstimator):
    """Aggregate an auxiliary dataframe before joining it on a base dataframe.

    Apply numerical and categorical aggregation operations on the columns (i.e. `cols`)
    to aggregate. See the list of supported operations at the parameter `operations`.

    If `cols` is not provided, `cols` are all columns from `aux_table`,
    except `aux_key`.

    Accepts :obj:`pandas.DataFrame` and :class:`polars.DataFrame` inputs.

    Parameters
    ----------
    aux_table : DataFrameLike or "X"
        Auxiliary dataframe to aggregate then join on the base table.
        The placeholder string "X" can be provided to perform
        self-aggregation on the input data.

    operations : str or iterable of str
        Aggregation operations to perform on the auxiliary table.

        Supported operations are "count", "mode", "min", "max", "sum", "median",
        "mean", "std". The operations "sum", "median", "mean", "std" are reserved
        to numeric type columns.

    key : str, default=None
        The column name to use for both `main_key` and `aux_key` when they
        are the same. Provide either `key` or both `main_key` and `aux_key`.
        If `key` is an iterable, we will perform a multi-column join.

    main_key : str or iterable of str, default=None
        Select the columns from the main table to use as keys during
        the join operation.
        If `main_key` is an iterable, we will perform a multi-column join.

    aux_key : str or iterable of str, default=None
        Select the columns from the auxiliary dataframe to use as keys during
        the join operation.
        If `aux_key` is an iterable, we will perform a multi-column join.

    cols : str or iterable of str or selector, default=s.all()
        Select the columns from the auxiliary dataframe to use as values during
        the aggregation operations.
        By default, `cols` are all columns from `aux_table`, except `aux_key`.

    suffix : str, default=""
        Suffix to append to the `aux_table`'s column names. You can use it
        to avoid duplicate column names in the join.

    See Also
    --------
    AggTarget :
        Aggregates the target `y` before joining its aggregation on the base dataframe.

    Joiner :
        Augments a main table by automatically joining an auxiliary table on it.

    MultiAggJoiner :
        Extension of the AggJoiner to multiple auxiliary tables.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub import AggJoiner
    >>> main = pd.DataFrame({
    ...     "airportId": [1, 2],
    ...     "airportName": ["Paris CDG", "NY JFK"],
    ... })
    >>> aux = pd.DataFrame({
    ...     "flightId": range(1, 7),
    ...     "from_airport": [1, 1, 1, 2, 2, 2],
    ...     "total_passengers": [90, 120, 100, 70, 80, 90],
    ...     "company": ["DL", "AF", "AF", "DL", "DL", "TR"],
    ... })
    >>> agg_joiner = AggJoiner(
    ...     aux_table=aux,
    ...     operations="mean",
    ...     main_key="airportId",
    ...     aux_key="from_airport",
    ...     cols="total_passengers",
    ... )
    >>> agg_joiner.fit_transform(main)
       airportId  airportName  total_passengers_mean
    0          1    Paris CDG              103.33...
    1          2       NY JFK               80.00...
    """

    def __init__(
        self,
        aux_table,
        operations,
        *,
        key=None,
        main_key=None,
        aux_key=None,
        cols=s.all(),
        suffix="",
    ):
        self.aux_table = aux_table
        self.operations = operations
        self.key = key
        self.main_key = main_key
        self.aux_key = aux_key
        self.cols = cols
        self.suffix = suffix

    def _check_inputs(self, X):
        """Check inputs before fitting.

        Parameters
        ----------
        X : DataFrameLike
            Input data, base table on which to left join the
            auxiliary table.
        """
        self._main_key, self._aux_key = _join_utils.check_key(
            self.main_key, self.aux_key, self.key
        )
        _join_utils.check_missing_columns(X, self._main_key, "'X' (the main table)")
        _join_utils.check_missing_columns(self._aux_table, self._aux_key, "'aux_table'")
        _join_utils.check_missing_columns(
            self._aux_table, s.make_selector(self.cols), ""
        )

        # If no `cols` provided, all columns but `aux_key` are used.
        self._cols = s.make_selector(self.cols) - self._aux_key

        self._operations = atleast_1d_or_none(self.operations)
        if (
            not all([isinstance(op, str) for op in self._operations])
            or self._operations == []
        ):
            raise ValueError(
                "`operations` must be string or an iterable of strings, got"
                f" {self.operations}."
            )

        unsupported_ops = set(self._operations).difference(set(SUPPORTED_OPS))
        if unsupported_ops:
            raise ValueError(
                f"`operations` options are {SUPPORTED_OPS}, but {unsupported_ops} are"
                " not supported."
            )

        if not isinstance(self.suffix, str):
            raise ValueError(f"'suffix' must be a string. Got {self.suffix}")

    def fit_transform(self, X, y=None):
        """Aggregate auxiliary table based on the main keys.

        Parameters
        ----------
        X : DataFrameLike
            Input data, based table on which to left join the
            auxiliary table.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        DataFrame
            The augmented input.
        """
        if isinstance(self.aux_table, str) and self.aux_table == "X":
            self.aux_table = X
        elif not sbd.is_dataframe(self.aux_table):
            raise ValueError(
                "'aux_table' must be a dataframe or the string 'X', got"
                f" {type(self.aux_table)}. If you have more than one 'aux_table',"
                " use the MultiAggJoiner instead."
            )
        self._aux_table = CheckInputDataFrame().fit_transform(self.aux_table)
        self._main_check_input = CheckInputDataFrame()
        X = self._main_check_input.fit_transform(X)
        self._check_inputs(X)

        self.aux_table_ = aggregate(
            self._aux_table,
            self._aux_key,
            self._cols.expand(self._aux_table),
            self._operations,
            suffix=self.suffix,
        )
        result = _join_utils.left_join(
            X,
            right=self.aux_table_,
            left_on=self._main_key,
            right_on=self._aux_key,
        )
        self.all_outputs_ = sbd.column_names(result)
        return result

    def fit(self, X, y=None):
        """Aggregate auxiliary table based on the main keys.

        Parameters
        ----------
        X : DataFrameLike
            Input data, based table on which to left join the
            auxiliary table.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        AggJoiner
            Fitted :class:`AggJoiner` instance (self).
        """
        _ = self.fit_transform(X, y)
        return self

    def transform(self, X):
        """Left-join pre-aggregated table on `X`.

        Parameters
        ----------
        X : DataFrameLike
            The input data to transform.

        Returns
        -------
        DataFrame
            The augmented input.
        """
        check_is_fitted(self, "aux_table_")
        X = self._main_check_input.transform(X)

        result = _join_utils.left_join(
            X,
            right=self.aux_table_,
            left_on=self._main_key,
            right_on=self._aux_key,
        )

        result = sbd.set_column_names(result, self.all_outputs_)
        return result

    def get_feature_names_out(self):
        """Get output feature names for transformation.

        Returns
        -------
        List of str
            Transformed feature names.
        """
        check_is_fitted(self, "aux_table_")
        return self.all_outputs_


class AggTarget(TransformerMixin, BaseEstimator):
    """Aggregate a target `y` before joining its aggregation on a base dataframe.

    Accepts :obj:`pandas.DataFrame` or :class:`polars.DataFrame` inputs.

    Parameters
    ----------
    main_key : str or iterable of str
        Select the columns from the main table to use as keys during
        the aggregation of the target and during the join operation.

        If `main_key` refer to a single column, a single aggregation
        for this key will be generated and a single join will be performed.

        If `main_key` is a list of keys, a multi-column aggregation will be performed
        on the target.

    operations : str or iterable of str
        Aggregation operations to perform on the target.

        Supported operations are "count", "mode", "min", "max", "sum", "median",
        "mean", "std". The operations "sum", "median", "mean", "std" are reserved
        to numeric type targets.

    suffix : str, default="_target"
        The suffix to append to the columns of the target table if the join
        results in duplicates columns.

    See Also
    --------
    AggJoiner :
        Aggregates auxiliary dataframes before joining them
        on the base dataframe.

    Joiner :
        Augments a main table by automatically joining multiple
        auxiliary tables on it.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from skrub import AggTarget
    >>> X = pd.DataFrame({
    ...     "flightId": range(1, 7),
    ...     "from_airport": [1, 1, 1, 2, 2, 2],
    ...     "total_passengers": [90, 120, 100, 70, 80, 90],
    ...     "company": ["DL", "AF", "AF", "DL", "DL", "TR"],
    ... })
    >>> y = np.array([1, 1, 0, 0, 1, 1])
    >>> agg_target = AggTarget(
    ...     main_key="company",
    ...     operations=["mean", "max"],
    ... )
    >>> agg_target.fit_transform(X, y)
       flightId  from_airport  ...  y_0_mean_target  y_0_max_target
    0         1             1  ...         0.666667               1
    1         2             1  ...         0.500000               1
    2         3             1  ...         0.500000               1
    3         4             2  ...         0.666667               1
    4         5             2  ...         0.666667               1
    5         6             2  ...         1.000000               1
    """

    def __init__(
        self,
        main_key,
        operations,
        *,
        suffix="_target",
    ):
        self.main_key = main_key
        self.operations = operations
        self.suffix = suffix

    def check_inputs(self, X, y):
        """Check inputs before fitting.

        Parameters
        ----------
        X : DataFrameLike
            Input data, base table on which to left join the target.
        y : DataFrameLike or SeriesLike or ArrayLike
            The target to check.

        Returns
        -------
        y_ : DataFrameLike
            The transformed target.
        """
        if not sbd.is_dataframe(X):
            raise TypeError(f"X must be a dataframe, got {type(X)}")

        self._main_key = atleast_1d_or_none(self.main_key)
        _join_utils.check_missing_columns(X, self._main_key, "'X' (the main table)")

        # `y` is copied or converted to a df to be compatible with `aggregate`

        # If `y` is already a dataframe
        if sbd.is_dataframe(y):
            y_ = sbd.make_dataframe_like(y, sbd.to_column_list(y))
        # If `y` is a named series, we convert it to a dataframe
        elif sbd.is_column(y) and sbd.name(y) is not None:
            y_ = sbd.make_dataframe_like(y, {sbd.name(y): y})
        else:
            # If `y` is an unnamed series
            if sbd.is_column(y):
                y = sbd.to_numpy(y).reshape(-1, 1)
            y_ = np.atleast_2d(y)

            cols = {f"y_{i}": y_[:, i] for i in range(y_.shape[1])}
            y_ = sbd.make_dataframe_like(X, cols)

        # Check lengths
        if sbd.shape(y_)[0] != sbd.shape(X)[0]:
            raise ValueError(
                f"X and y length must match, got {sbd.shape(X)[0]} and"
                f" {sbd.shape(y_)[0]}."
            )

        self._cols = sbd.column_names(y_)

        self._operations = atleast_1d_or_none(self.operations)
        if (
            not all([isinstance(op, str) for op in self._operations])
            or self._operations == []
        ):
            raise ValueError(
                "`operations` must be string or an iterable of strings, got"
                f" {self.operations}."
            )

        unsupported_ops = set(self._operations).difference(set(SUPPORTED_OPS))
        if unsupported_ops:
            raise ValueError(
                f"`operations` options are {SUPPORTED_OPS}, but {unsupported_ops} are"
                " not supported."
            )

        if not isinstance(self.suffix, str):
            raise ValueError(f"'suffix' must be a string, got {self.suffix!r}")

        return y_

    def fit_transform(self, X, y):
        """Aggregate the target `y` based on keys from `X`.

        Parameters
        ----------
        X : DataFrameLike
            Must contains the columns names defined in `main_key`.

        y : DataFrameLike or SeriesLike or ArrayLike
            `y` length must match `X` length.
            The target can be continuous or discrete, with multiple columns.

        Returns
        -------
        Dataframe
            The augmented input.
        """
        y_ = self.check_inputs(X, y)
        self._main_check_input = CheckInputDataFrame()
        X = self._main_check_input.fit_transform(X)

        # Add the main key on the target
        y_ = sbd.with_columns(y_, **{k: sbd.col(X, k) for k in self._main_key})

        self.y_ = aggregate(
            y_,
            key=self._main_key,
            cols_to_agg=self._cols,
            operations=self._operations,
            suffix=self.suffix,
        )

        result = _join_utils.left_join(
            X,
            right=self.y_,
            left_on=self._main_key,
            right_on=self._main_key,
        )
        self.all_outputs_ = sbd.column_names(result)
        return result

    def fit(self, X, y):
        """Aggregate the target `y` based on keys from `X`.

        Parameters
        ----------
        X : DataFrameLike
            Must contains the columns names defined in `main_key`.

        y : DataFrameLike or SeriesLike or ArrayLike
            `y` length must match `X` length.
            The target can be continuous or discrete, with multiple columns.

        Returns
        -------
        AggTarget
            Fitted :class:`AggTarget` instance (self).
        """
        _ = self.fit_transform(X, y)
        return self

    def transform(self, X):
        """Left-join pre-aggregated target on `X`.

        Parameters
        ----------
        X : DataFrameLike
            The input data to transform.

        Returns
        -------
        X_transformed : DataFrameLike
            The augmented input.
        """
        check_is_fitted(self, "y_")
        X = self._main_check_input.transform(X)

        result = _join_utils.left_join(
            X,
            right=self.y_,
            left_on=self._main_key,
            right_on=self._main_key,
        )
        result = sbd.set_column_names(result, self.all_outputs_)
        return result

    def get_feature_names_out(self):
        """Get output feature names for transformation.

        Returns
        -------
        List of str
            Transformed feature names.
        """
        check_is_fitted(self, "y_")
        return self.all_outputs_
