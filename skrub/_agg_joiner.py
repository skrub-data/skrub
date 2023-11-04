"""
Implement AggJoiner and AggTarget to join a main table to its auxiliary tables,
with one-to-many relationships.

Both classes aggregate the auxiliary tables first, then join these grouped
tables with the base table.
"""
from copy import deepcopy
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from skrub._dataframe._namespace import get_df_namespace
from skrub._dataframe._pandas import _parse_argument
from skrub._utils import atleast_1d_or_none, atleast_2d_or_none

NUM_OPERATIONS = ["sum", "mean", "std", "min", "max", "hist", "value_counts"]
CATEG_OPERATIONS = ["mode", "count", "value_counts"]
ALL_OPS = NUM_OPERATIONS + CATEG_OPERATIONS


def split_num_categ_operations(operations: list[str]) -> tuple[list[str], list[str]]:
    """Separate aggregagor operators input by their type.

    Parameters
    ----------
    operations : list of str
        The input operators names.

    Returns
    -------
    num_operations, categ_operations : Tuple of List of str
        List of operator names
    """
    num_operations, categ_operations = [], []
    for operation in operations:
        # hist(5) -> hist
        op_root, _ = _parse_argument(operation)
        if op_root in NUM_OPERATIONS:
            num_operations.append(operation)
        if op_root in CATEG_OPERATIONS:
            categ_operations.append(operation)
        if op_root not in ALL_OPS:
            raise ValueError(f"operations options are {ALL_OPS}, got: {operation=!r}.")

    return num_operations, categ_operations


def check_missing_columns(
    X,
    columns,
    error_msg,
):
    """All elements of main_key must belong to the columns of X.

    Parameters
    ----------
    X : DataFrameLike
        Input data.

    main_key : list of string
        Key used to perform join on X.
    """
    missing_cols = set(columns) - set(X.columns)
    if len(missing_cols) > 0:
        raise ValueError(error_msg)
    return


class AggJoiner(BaseEstimator, TransformerMixin):
    """Aggregate auxiliary dataframes before joining them on a base dataframe.

    Apply numerical and categorical aggregation operations on the columns
    to aggregate, selected by dtypes. See the list of supported operations
    at the parameter `agg_ops`.

    The grouping columns used during the aggregation are the columns used
    as keys for joining.

    Accepts :obj:`pandas.DataFrame` and :class:`polars.DataFrame` inputs.

    Parameters
    ----------
    aux_table : DataFrameLike or str or iterable
        Auxiliary dataframe to aggregate then join on the base table.
        The placeholder string "X" can be provided to perform
        self-aggregation on the input data.

    aux_key : str, or iterable of str, or iterable of iterable of str
        Select the columns from the auxiliary dataframe to use as keys during
        the join operation.

    main_key : str or iterable of str
        Select the columns from the main table to use as keys during
        the join operation.
        If main_key is a list, we will perform a multi-column join.

    cols : str, or iterable of str, or iterable of iterable of str, default=None
        Select the columns from the auxiliary dataframe to use as values during
        the aggregation operations.
        If None, cols are all columns from table, except `aux_key`.

    operation : str or iterable of str, default=None
        Aggregation operations to perform on the auxiliary table.

        numerical : {"sum", "mean", "std", "min", "max", "hist", "value_counts"}
            'hist' and 'value_counts' accepts an integer argument to parametrize
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
    AggTarget :
        Aggregates the target `y` before joining its aggregation
        on the base dataframe.

    Joiner :
        Augments a main table by automatically joining multiple
        auxiliary tables on it.

    Examples
    --------
    >>> import pandas as pd
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
    >>> join_agg = AggJoiner(
    ...     aux_table=aux,
    ...     aux_key="from_airport",
    ...     main_key="airportId",
    ...     cols=["total_passengers", "company"],
    ...     operation=["mean", "mode"],
    ... )
    >>> join_agg.fit_transform(main)
       airportId airportName company_mode_1  total_passengers_mean_1
    0          1   Paris CDG             AF               103.33...
    1          2      NY JFK             DL                80.00...
    """

    def __init__(
        self,
        aux_table,
        *,
        aux_key,
        main_key,
        cols=None,
        operation=None,
        suffix=None,
    ):
        self.aux_table = aux_table
        self.aux_key = aux_key
        self.cols = cols
        self.main_key = main_key
        self.operation = operation
        self.suffix = suffix

    def fit(self, X, y=None):
        """Aggregate auxiliary tables based on the main keys.

        Parameters
        ----------
        X : DataframeLike
            Input data, based table on which to left join the
            auxiliary tables.

        y : array-like of shape (n_samples), default=None
            Prediction target. Used to compute correlations between the
            generated covariates and the target for screening purposes.

        Returns
        -------
        AggJoiner
            Fitted :class:`AggJoiner` instance (self).
        """
        self.check_input(X)
        skrub_px, _ = get_df_namespace(*self.aux_table_)

        num_operations, categ_operations = split_num_categ_operations(self.operation_)

        aux_tables = []
        for aux_table, aux_key, cols, suffix in zip(
            self.aux_table_, self.aux_key_, self.cols_, self.suffix_
        ):
            aux_table = skrub_px.aggregate(
                aux_table,
                aux_key,
                cols,
                num_operations,
                categ_operations,
                suffix=suffix,
            )
            aux_table = self._screen(aux_table, y)
            aux_tables.append((aux_table, aux_key))
        self.aux_table_ = aux_tables

        return self

    def transform(self, X):
        """Left-join pre-aggregated tables on `X`.

        Parameters
        ----------
        X : DataFrameLike
            The input data to transform.

        Returns
        -------
        X_transformed : DataFrameLike
            The augmented input.
        """

        check_is_fitted(self, "aux_table_")
        skrub_px, _ = get_df_namespace(*[aux_table for aux_table, _ in self.aux_table_])

        for aux_table, aux_key in self.aux_table_:
            X = skrub_px.join(
                left=X,
                right=aux_table,
                left_on=self.main_key_,
                right_on=aux_key,
            )

        return X

    def _screen(self, aux_table, y):
        """Only keep aggregated features which correlation with
        y is above some threshold.
        """
        # TODO: Add logic
        return aux_table

    def check_input(self, X):
        """Perform a check on column names data type and suffixes.

        Parameters
        ----------
        X : DataFrameLike
            The raw input to check.
        """
        # Polars lazyframes will raise an error here.
        if not hasattr(X, "__dataframe__"):
            raise TypeError(f"X must be a dataframe, got {type(X)}.")

        self.main_key_ = atleast_1d_or_none(self.main_key)
        self.suffix_ = atleast_1d_or_none(self.suffix)
        self.aux_key_ = atleast_2d_or_none(self.aux_key)
        self.cols_ = atleast_2d_or_none(self.cols)

        # Check main_key
        error_msg = f"main_key={self.main_key_!r} are not in {X.columns=!r}."
        check_missing_columns(X, self.main_key_, error_msg=error_msg)

        # Check length of table and aux_key
        if not isinstance(self.aux_table, (list, tuple)):
            tables = [self.aux_table]
        else:
            tables = self.aux_table

        if len(self.suffix_) == 0:
            self.suffix_ = [f"_{idx+1}" for idx in range(len(tables))]

        # Check tables and list of suffix match
        if len(tables) != len(self.suffix_):
            raise ValueError(
                "'suffix' must be None or match the "
                f"number of tables, got: {self.suffix_!r}"
            )

        # Check tables and list of aux_keys match
        if len(tables) != len(self.aux_key_):
            error_msg = (
                "The number of tables must match the number of aux_key, "
                f"got {len(tables)=!r} and {len(self.aux_key_)=!r}. "
            )
            if len(tables) > 1:
                error_msg += (
                    "For multiple tables, use a list of list, "
                    "e.g. aux_key=[['col1', 'col2'], ['colA', 'colB']]."
                )
            raise ValueError(error_msg)

        # Check table type and missing columns
        for idx, (table, aux_key, cols) in enumerate(
            zip(tables, self.aux_key_, self.cols_)
        ):
            if isinstance(table, str):
                if table != "X":
                    raise ValueError(
                        "If the dataframe is declared with a string, "
                        f"the only acceptable value is 'X', got {table!r}."
                    )
                table = deepcopy(X)
                tables[idx] = table

            elif not hasattr(table, "__dataframe__"):
                raise TypeError(
                    "'tables' must be a list of tuple and the first element of each"
                    f" tuple must be a dataFrame, got {type(tables[0])} at index"
                    f" {idx}."
                )

            # If no cols provided, all columns but aux_key are used.
            if len(cols) == 0:
                cols = list(set(table.columns) - set(aux_key))
                self.cols_[idx] = cols

            error_msg = f"{aux_key=!r} are not in {table.columns=!r}."
            check_missing_columns(table, aux_key, error_msg=error_msg)

            error_msg = f"{cols=!r} are not in {table.columns=!r}."
            check_missing_columns(table, cols, error_msg=error_msg)

            if len(aux_key) != len(self.main_key_):
                raise ValueError(
                    "The number of keys to join must match, got "
                    f"main_key={self.main_key_!r} and "
                    f"{aux_key=!r} for the table at index {idx}."
                )

        self.aux_table_ = tables

        # Check tables and list of cols match
        if len(tables) != len(self.cols_):
            error_msg = (
                "The number of tables must match the number of cols, "
                f"got {len(tables)=!r} and {len(self.cols_)=!r}. "
            )
            if len(tables) > 1:
                error_msg += (
                    "For multiple tables, use a list of list, "
                    "e.g. cols=[['col1'], ['colA']]."
                )
            raise ValueError(error_msg)

        # Check operation
        if self.operation is None:
            operation = ["mean", "mode"]
        else:
            operation = np.atleast_1d(self.operation).tolist()
        self.operation_ = operation

        return


class AggTarget(BaseEstimator, TransformerMixin):
    """Aggregate a target ``y`` before joining its aggregation on a base dataframe.

    Accepts :obj:`pandas.DataFrame` or :class:`polars.DataFrame` inputs.

    Parameters
    ----------
    main_key : str or iterable of str
        Select the columns from the main table to use as keys during
        the aggregation of the target and during the join operation.

        If main_key refer to a single column, a single aggregation
        for this key will be generated and a single join will be performed.

        Otherwise, if main_key is a list of keys, the target will be
        aggregated using each key separately, then each aggregation of
        the target will be joined on the main table.

    operation : str or iterable of str, optional
        Aggregation operations to perform on the auxiliary table.

        numerical : {"sum", "mean", "std", "min", "max", "hist(3)", "value_counts"}
            'hist' and 'value_counts' accepts an integer argument to parametrize
            the binning.

        categorical : {"mode", "count", "value_counts"}

        If set to None (the default), ['mean', 'mode'] will be used.

    suffix : str, optional
        The suffix to append to the columns of the target table if the join
        result in some duplicates columns.
        If set to None, "_target" is used.

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
    >>> X = pd.DataFrame({
    ...     "flightId": range(1, 7),
    ...     "from_airport": [1, 1, 1, 2, 2, 2],
    ...     "total_passengers": [90, 120, 100, 70, 80, 90],
    ...     "company": ["DL", "AF", "AF", "DL", "DL", "TR"],
    ... })
    >>> y = np.array([1, 1, 0, 0, 1, 1])
    >>> agg_target = AggTarget(
    ...     main_key="company",
    ...     operation=["mean", "max"],
    ... )
    >>> agg_target.fit_transform(X, y)
       flightId  from_airport  ...  y_0_max_target y_0_mean_target
    0         1             1  ...               1        0.666667
    1         2             1  ...               1        0.500000
    2         3             1  ...               1        0.500000
    3         4             2  ...               1        0.666667
    4         5             2  ...               1        0.666667
    5         6             2  ...               1        1.000000
    <BLANKLINE>
    [6 rows x 6 columns]
    """

    def __init__(
        self,
        main_key: str | Iterable[str],
        operation: str | Iterable[str] | None = None,
        suffix: str | None = None,
    ):
        self.main_key = main_key
        self.operation = operation
        self.suffix = suffix

    def fit(self, X, y):
        """Aggregate the target ``y`` based on keys from ``X``.

        Parameters
        ----------
        X : DataFrameLike
            Must contains the columns names defined in ``main_key``.

        y : DataFrameLike or SeriesLike or ArrayLike
            ``y`` length must match ``X`` length, with matching indices.
            The target can be continuous or discrete, with multiple columns.

            If the target is continuous, only numerical operations,
            listed in ``num_operations``, can be applied.

            If the target is discrete, only categorical operations,
            listed in ``categ_operations``, can be applied.

            Note that the target type is determined by
            :func:`sklearn.utils.multiclass.type_of_target`.

        Returns
        -------
        AggTarget
            Fitted AggTarget instance (self).
        """
        y_ = self.check_input(X, y)
        skrub_px, _ = get_df_namespace(X, y_)

        # Add the main key on the target
        y_[self.main_key_] = X[self.main_key_]

        num_operations, categ_operations = split_num_categ_operations(self.operation_)

        self.y_ = skrub_px.aggregate(
            y_,
            key=self.main_key_,
            cols_to_agg=self.cols_,
            num_operations=num_operations,
            categ_operations=categ_operations,
            suffix=self.suffix_,
        )

        return self

    def transform(self, X):
        """Left-join pre-aggregated tables on `X`.

        Parameters
        ----------
        X : DataFrameLike,
            The input data to transform.

        Returns
        -------
        X_transformed : DataFrameLike,
            The augmented input.
        """
        check_is_fitted(self, "y_")
        skrub_px, _ = get_df_namespace(X)

        return skrub_px.join(
            left=X,
            right=self.y_,
            left_on=self.main_key_,
            right_on=self.main_key_,
        )

    def check_input(self, X, y):
        """Perform a check on column names data type and suffixes.

        Parameters
        ----------
        X : DataFrameLike
            The raw input to check.
        y : DataFrameLike or SeriesLike or ArrayLike
            The raw target to check.

        Returns
        -------
        y_ : DataFrameLike,
            Transformation of the target.
        """
        if not hasattr(X, "__dataframe__"):
            raise TypeError(f"X must be a dataframe, got {type(X)}")

        self.main_key_ = atleast_1d_or_none(self.main_key)

        self.suffix_ = "_target" if self.suffix is None else self.suffix

        if not isinstance(self.suffix_, str):
            raise ValueError(f"'suffix' must be a string, got {self.suffix_!r}")

        error_msg = f"{self.main_key_=!r} not in {X.columns=!r}"
        check_missing_columns(X, self.main_key_, error_msg=error_msg)

        # If y is not a dataframe, we convert it.
        if hasattr(y, "__dataframe__"):
            # Need to copy since we add columns in place
            # during fit.
            y_ = y.copy()
        else:
            y_ = np.atleast_2d(y)

            # If y is Series or an array derived from a
            # Series, we need to transpose it.
            if len(y_) == 1 and len(y) != 1:
                y_ = y_.T

            _, px = get_df_namespace(X)
            y_ = px.DataFrame(y_)

            if hasattr(y, "name"):
                # y is a Series
                cols = [y.name]
            else:
                cols = [f"y_{col}" for col in y_.columns]
            y_.columns = cols

        # Check lengths
        if y_.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y length must match, got {X.shape[0]} and {y_.shape[0]}."
            )

        self.cols_ = y_.columns

        if self.operation is None:
            y_type = type_of_target(y_)
            if y_type in ["continuous", "continuous-multioutput"]:
                operation = ["mean"]
            else:
                operation = ["mode"]
        else:
            operation = np.atleast_1d(self.operation).tolist()
        self.operation_ = operation

        return y_
