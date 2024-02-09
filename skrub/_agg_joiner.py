"""
Implement AggJoiner and AggTarget to join a main table to an auxiliary table,
with one-to-many relationships.

Both classes aggregate the auxiliary table first, then join this grouped
table with the main table.
"""
from collections import Counter
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from skrub import _join_utils
from skrub._dataframe._namespace import get_df_namespace, is_pandas, is_polars
from skrub._dataframe._pandas import _parse_argument
from skrub._utils import atleast_1d_or_none

NUM_OPERATIONS = ["sum", "mean", "std", "min", "max", "hist", "value_counts"]
CATEG_OPERATIONS = ["mode", "count", "value_counts"]
ALL_OPS = NUM_OPERATIONS + CATEG_OPERATIONS


def split_num_categ_operations(operations: list[str]) -> tuple[list[str], list[str]]:
    """Separate aggregator operators input by their type.

    Parameters
    ----------
    operations : list of str
        The input operators names.

    Returns
    -------
    num_operations, categ_operations : Tuple of list of str
        List of operator names.
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


class AggJoiner(BaseEstimator, TransformerMixin):
    """Aggregate an auxiliary dataframe before joining it on a base dataframe.

    Apply numerical and categorical aggregation operations on the `cols`
    to aggregate, selected by dtypes. See the list of supported operations
    at the parameter `operation`.

    If `cols` is not provided, `cols` are all columns from `aux_table`,
    except `aux_key`.

    Accepts :obj:`pandas.DataFrame` and :class:`polars.DataFrame` inputs.

    Parameters
    ----------
    aux_table : DataFrameLike or "X"
        Auxiliary dataframe to aggregate then join on the base table.
        The placeholder string "X" can be provided to perform
        self-aggregation on the input data.

    key : str, default=None
        The column names to use for both `main_key` and `aux_key` when they
        are the same. Provide either `key` or both `main_key` and `aux_key`.

    main_key : str or iterable of str, default=None
        Select the columns from the main table to use as keys during
        the join operation.
        If `main_key` is a list, we will perform a multi-column join.

    aux_key : str or iterable of str, default=None
        Select the columns from the auxiliary dataframe to use as keys during
        the join operation.

    cols : str or iterable of str, default=None
        Select the columns from the auxiliary dataframe to use as values during
        the aggregation operations.
        If set to `None`, `cols` are all columns from `aux_table`, except `aux_key`.

    operation : str or iterable of str, default=None
        Aggregation operations to perform on the auxiliary table.

        - numerical : {"sum", "mean", "std", "min", "max", "hist", "value_counts"}
        "hist" and "value_counts" accept an integer argument to parametrize
        the binning.

        - categorical : {"mode", "count", "value_counts"}

        - If set to `None` (the default), ["mean", "mode"] will be used.

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
    ...     main_key="airportId",
    ...     aux_key="from_airport",
    ...     cols=["total_passengers", "company"],
    ...     operation=["mean", "mode"],
    ... )
    >>> agg_joiner.fit_transform(main)
       airportId  airportName  company_mode  total_passengers_mean
    0          1    Paris CDG            AF              103.33...
    1          2       NY JFK            DL               80.00...
    """

    def __init__(
        self,
        aux_table,
        *,
        key=None,
        main_key=None,
        aux_key=None,
        cols=None,
        operation=None,
        suffix="",
    ):
        self.aux_table = aux_table
        self.key = key
        self.main_key = main_key
        self.aux_key = aux_key
        self.cols = cols
        self.operation = operation
        self.suffix = suffix

    def _check_dataframes(self, X, aux_table):
        """Check dataframes input types.

        Raises an error if frames aren't both Pandas or Polars dataframes,
        or if there is a Polars lazyframe. Alternatively, allows `aux_table` to be "X".

        Parameters
        ----------
        X : DataframeLike
            The main table to augment.
        aux_table : DataframeLike or "X"
            The auxiliary table.
        """
        if isinstance(aux_table, str):
            if aux_table == "X":
                return X, X
            else:
                raise ValueError("'aux_table' must be a dataframe or 'X'.")
        # Polars lazyframes will raise an error here.
        if not hasattr(X, "__dataframe__"):
            raise TypeError(f"'X' must be a dataframe, got {type(X)}.")
        if not hasattr(aux_table, "__dataframe__"):
            raise TypeError(
                f"'aux_table' must be a dataframe or 'X', got {type(aux_table)}."
            )

        if (is_pandas(X) and not is_pandas(aux_table)) or (
            is_polars(X) and not is_polars(aux_table)
        ):
            raise TypeError(
                "'X' and 'aux_table' must be of the same type, got"
                f"{type(X)} and {type(aux_table)}"
            )

        return X, aux_table

    def _check_cols(self):
        """Check `cols` to aggregate.

        If None, `cols` are all columns from `aux_table`, except `aux_key`.

        Returns
        -------
        cols
            1-dimensional array of columns on which to perform aggregation.
        """
        cols = atleast_1d_or_none(self.cols)
        # If no cols provided, all columns but `aux_key` are used.
        if len(cols) == 0:
            cols = list(set(self._aux_table.columns) - set(self._aux_key))
        if not all([col in self._aux_table.columns for col in cols]):
            raise ValueError("All 'cols' must be present in 'aux_table'.")
        return cols

    def _check_operation(self):
        """Check operation input.

        Returns
        -------
        operation
            1-dimensional array of operations to perform on columns.
        """
        operation = atleast_1d_or_none(self.operation)
        if len(operation) == 0:
            operation = ["mean", "mode"]
        return operation

    def _check_suffix(self):
        if not isinstance(self.suffix, str):
            raise ValueError(f"'suffix' must be a string. Got {self.suffix}")

    def _check_column_name_duplicates_after_aggregation(
        self,
        main_table,
        aux_table,
        main_key,
        aux_key,
        main_table_name="main_table",
        aux_table_name="aux_table",
    ):
        """Check that there are no duplicate column names after aggregating.

        Parameters
        ----------
        main_table : dataframe
            The main table to join.
        aux_table : dataframe
            The aggregated auxiliary table to join.
        main_table_name : str
            How to refer to ``main_table`` in error messages.
        aux_table_name : str
            How to refer to ``aux_table`` in error messages.
        Raises
        ------
        ValueError
            If any of the table has duplicate column names, or if there are column
            names that are used in both tables.
        """
        main_columns = list(set(main_table.columns) - set(main_key))
        aux_columns = [f"{col}" for col in list(set(aux_table.columns) - set(aux_key))]
        for columns, table_name in [
            (main_columns, main_table_name),
            (aux_columns, aux_table_name),
        ]:
            counts = Counter(columns)
            duplicates = [k for k, v in counts.items() if v > 1]
            if duplicates:
                raise ValueError(
                    f"Table '{table_name}' has duplicate column names: {duplicates}."
                    " Please make sure column names are unique."
                )
        overlap = list(set(main_columns).intersection(aux_columns))
        if overlap:
            raise ValueError(
                "After aggregating the auxiliary table, the following"
                f" column names are found in both tables '{main_table_name}' and"
                f" '{aux_table_name}': {overlap}. Please make sure column names do not"
                " overlap by renaming some columns or choosing a different suffix."
            )

    def _check_input(self, X):
        """Check inputs before fitting.

        Parameters
        ----------
        X : DataframeLike
            Input data, based table on which to left join the
            auxiliary table.
        """
        X, self._aux_table = self._check_dataframes(X, self.aux_table)
        self._main_key, self._aux_key = _join_utils.check_key(
            self.main_key, self.aux_key, self.key
        )
        _join_utils.check_missing_columns(X, self._main_key, "'X' (the main table)")
        _join_utils.check_missing_columns(self._aux_table, self._aux_key, "'aux_table'")

        # self._cols = self._check_cols()
        self._cols = atleast_1d_or_none(self.cols)
        # If no cols provided, all columns but `aux_key` are used.
        if len(self._cols) == 0:
            self._cols = list(set(self._aux_table.columns) - set(self._aux_key))
        _join_utils.check_missing_columns(self._aux_table, self._cols, "'aux_table'")

        self._operation = self._check_operation()
        self.num_operations, self.categ_operations = split_num_categ_operations(
            self._operation
        )
        self._check_suffix()

    def fit(self, X, y=None):
        """Aggregate auxiliary table based on the main keys.

        Parameters
        ----------
        X : DataframeLike
            Input data, based table on which to left join the
            auxiliary table.

        y : array-like of shape (n_samples), default=None
            Prediction target. Used to compute correlations between the
            generated covariates and the target for screening purposes.

        Returns
        -------
        AggJoiner
            Fitted :class:`AggJoiner` instance (self).
        """
        self._check_input(X)
        skrub_px, _ = get_df_namespace(self._aux_table)
        aux_table = skrub_px.aggregate(
            self._aux_table,
            self._aux_key,
            self._cols,
            self.num_operations,
            self.categ_operations,
            suffix=self.suffix,
        )
        self.aux_table_ = self._screen(aux_table, y)
        self._check_column_name_duplicates_after_aggregation(
            X, self.aux_table_, self._main_key, self._aux_key, main_table_name="X"
        )
        return self

    def transform(self, X):
        """Left-join pre-aggregated table on `X`.

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
        X, _ = self._check_dataframes(X, self.aux_table_)
        _join_utils.check_missing_columns(X, self._main_key, "'X' (the main table)")

        skrub_px, _ = get_df_namespace(self.aux_table_)
        X = skrub_px.join(
            left=X,
            right=self.aux_table_,
            left_on=self._main_key,
            right_on=self._aux_key,
        )

        return X

    def _screen(self, aux_table, y):
        """Only keep aggregated features which correlation with
        y is above some threshold.
        """
        # TODO: Add logic
        return aux_table


class AggTarget(BaseEstimator, TransformerMixin):
    """Aggregate a target ``y`` before joining its aggregation on a base dataframe.

    Accepts :obj:`pandas.DataFrame` or :class:`polars.DataFrame` inputs.

    Parameters
    ----------
    main_key : str or iterable of str
        Select the columns from the main table to use as keys during
        the aggregation of the target and during the join operation.

        If `main_key` refer to a single column, a single aggregation
        for this key will be generated and a single join will be performed.

        Otherwise, if `main_key` is a list of keys, the target will be
        aggregated using each key separately, then each aggregation of
        the target will be joined on the main table.

    operation : str or iterable of str, optional
        Aggregation operations to perform on the target.

        numerical : {"sum", "mean", "std", "min", "max", "hist", "value_counts"}
            'hist' and 'value_counts' accept an integer argument to parametrize
            the binning.

        categorical : {"mode", "count", "value_counts"}

        If set to None (the default), ["mean", "mode"] will be used.

    suffix : str, optional
        The suffix to append to the columns of the target table if the join
        results in duplicates columns.
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
            Must contains the columns names defined in `main_key`.

        y : DataFrameLike or SeriesLike or ArrayLike
            `y` length must match `X` length, with matching indices.
            The target can be continuous or discrete, with multiple columns.

            If the target is continuous, only numerical operations,
            listed in `num_operations`, can be applied.

            If the target is discrete, only categorical operations,
            listed in `categ_operations`, can be applied.

            Note that the target type is determined by
            :func:`sklearn.utils.multiclass.type_of_target`.

        Returns
        -------
        AggTarget
            Fitted :class:`AggTarget` instance (self).
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
        """Left-join pre-aggregated table on `X`.

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

        _join_utils.check_missing_columns(X, self.main_key_, "'X' (the main table)")

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
