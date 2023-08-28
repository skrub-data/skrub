"""
Implement AggJoiner and AggTarget to join a main table to its auxiliary tables,
with one-to-many relationships.

Both classes aggregate the auxiliary tables first, then join these grouped
tables with the base table.
"""
import sys
from copy import deepcopy
from types import ModuleType

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

import skrub.agg_joiner._agg_pandas as skrub_pd
import skrub.agg_joiner._agg_polars as skrub_pl
from skrub._utils import DataFrameLike, SeriesLike

NUM_OPERATIONS = ["sum", "mean", "std", "min", "max", "hist", "value_counts"]
CATEG_OPERATIONS = ["mode", "count", "value_counts"]
ALL_OPS = NUM_OPERATIONS + CATEG_OPERATIONS


def split_num_categ_operations(operations: list[str]) -> tuple[list[str], list[str]]:
    """Separate aggregagor operators input by their type.

    Parameters
    ----------
    operations : list of str,
        The input operators names.

    Returns
    -------
    num_operations, categ_operations : Tuple of List of str
        List of operator names
    """
    num_operations, categ_operations = [], []
    for operation in operations:
        # hist(5) -> hist
        op_root = operation.split("(")[0]
        if op_root in NUM_OPERATIONS:
            num_operations.append(operation)
        if op_root in CATEG_OPERATIONS:
            categ_operations.append(operation)
        if op_root not in ALL_OPS:
            raise ValueError(f"operations options are {ALL_OPS}, got: {operation=!r}.")

    return num_operations, categ_operations


def _is_pandas(dataframe: DataFrameLike) -> bool:
    return isinstance(dataframe, pd.DataFrame)


def _is_polars(dataframe: DataFrameLike) -> bool:
    if "polars" not in sys.modules:
        return False

    import polars as pl

    return isinstance(dataframe, (pl.DataFrame, pl.LazyFrame))


def get_df_namespace(*dfs: list[DataFrameLike]) -> tuple[ModuleType, ModuleType]:
    """Get the namespaces of dataframes.

    Introspect `dataframes` arguments and return their skrub namespace object
    skrub.agg_joiner._agg_{pandas, polars} and the dataframe module
    {polars, pandas} itself.

    The modules of input dataframes need to be the same, otherwise a TypeError
    is raised.

    The outputs of this function are denoted skrub_px and px in reference to
    the array API, returning namespace (numpy, pytorch and cupy) as ``n```.
    Since we deal with Polars (``pl``) and Pandas (``pd``), we use ``px``
    as a variable name.

    Parameters
    ----------
    dfs : list[DataFrameLike],
        The dataframes to extract modules from.

    Returns
    -------
    skrub_px: ModuleType
        Skrub namespace shared by dataframe objects.

    px : ModuleType
        Dataframe namespace, i.e. Pandas or Polars module.
    """
    # FIXME Pandas and Polars series will raise errors.
    if all([_is_pandas(df) for df in dfs]):
        return skrub_pd, pd

    elif all([_is_polars(df) for df in dfs]):
        import polars as pl

        if all([isinstance(df, pl.DataFrame) for df in dfs]) or all(
            [isinstance(df, pl.LazyFrame) for df in dfs]
        ):
            return skrub_pl, pl
        else:
            raise TypeError("Mixing polars lazyframes and dataframes is not supported.")

    else:
        modules = [type(df).__module__ for df in dfs]
        if all([_is_polars(df) or _is_pandas(df) for df in dfs]):
            raise TypeError(
                "Mixing Pandas and Polars dataframes is not supported, "
                f"got {modules=!r}."
            )
        else:
            raise TypeError(
                "Only Pandas or Polars dataframes are currently supported, "
                f"got {modules=!r}."
            )


def check_missing_columns(X: DataFrameLike, main_keys: list[str]) -> list[str]:
    """All elements of main_key must belong to the columns of X.

    Parameters
    ----------
    X : DataFrameLike,
        Input data.

    main_keys : list of string,
        Key used to perform join on X.

    Returns
    -------
    main_keys : list of string
    """
    main_keys = np.atleast_1d(main_keys).tolist()
    missing_cols = set(main_keys) - set(X.columns)
    if len(missing_cols) > 0:
        raise ValueError(
            f"Got {main_keys=!r}, but these columns are not in {X.columns=}."
        )
    return main_keys


def check_suffixes(suffixes: list[str], n_aux_tables: int) -> list[str]:
    """Check the length of suffixes match the number of tables.

    If suffixes is None, we use the range ``[1, n_aux_table+1]``.

    Parameters
    ----------
    suffixes : str or list of string
        Suffixes to be appended to column names, for each table.
    n_aux_tables : int,
        Number of auxiliary tables.

    Returns
    -------
    suffixes : list of string
        Cleaned list of suffix.
    """
    if suffixes is None:
        if n_aux_tables == 1:
            suffixes = [""]
        else:
            suffixes = [f"_{idx+1}" for idx in range(n_aux_tables)]
    elif isinstance(suffixes, str):
        suffixes = [suffixes]
    elif hasattr(suffixes, "__len__"):
        if len(suffixes) != n_aux_tables:
            raise ValueError(
                "Suffixes must be None or match the "
                f"number of auxiliary tables, got: {suffixes!r}"
            )
        suffixes = list(suffixes)
    else:
        raise ValueError(
            "Suffixes must be a list of string matching "
            f"the number of tables, got: {suffixes!r}"
        )
    return suffixes


class AggJoiner(BaseEstimator, TransformerMixin):
    """Aggregates auxiliary dataframes before joining them on the base dataframe.

    Apply numerical and categorical aggregation operations on the columns
    to aggregate, selected by dtypes. See the list of supported operations
    at the parameter `agg_ops`.

    The grouping columns used during the aggregation are the columns used
    as keys for joining.

    Uses :obj:`~pandas.DataFrame` and :obj:`~polars.DataFrame` inputs.

    Parameters
    ----------
    tables : list of tuples
        List of (dataframe, columns_to_join, columns_to_agg) tuple
        specifying the auxiliary dataframes and their columns for joining
        and aggregation operations.

        dataframe : DataFrameLike or str
            The auxiliary data to aggregate and join.
            The placeholder string "X" can be provided to perform
            self-aggregation on the input data.

        columns_to_join : str or array-like
            Select the columns from the auxiliary dataframe to use as keys during
            the join operation.

        columns_to_agg : str or array-like, optional
            Select the columns from the auxiliary dataframe to use as values during
            the aggregation operations.
            If missing, use all columns except `columns_to_join`.

    main_keys : str or array-like
        Select the columns from the main table to use as keys during
        the join operation.

        If main_keys is a list, we will perform a multi-column join.

    operations : str or list of str, optional
        Aggregation operations to perform on the auxiliary table.

        numerical : {"sum", "mean", "std", "min", "max", "hist", "value_counts"}
            'hist' and 'value_counts' accepts an integer argument to parametrize
            the binning.

        categorical : {"mode", "count", "value_counts"}

        If set to None (the default), ['mean', 'mode'] will be used.

    suffixes : list of str, default=None
        The suffixes that will be add to each table columns in case of
        duplicate column names, similar to `("_x", "_y")` in :obj:`~pandas.merge`.

        If set to None, we will use the table index, by looking at their
        order in 'tables', e.g. for a duplicate columns: price (main table),
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
    >>> main = pd.DataFrame({
            "airportId": [1, 2],
            "airportName": ["Paris CDG", "NY JFK"],
        })
    >>> aux = pd.DataFrame({
            "flightId": range(1, 7),
            "from_airport": [1, 1, 1, 2, 2, 2],
            "total_passengers": [90, 120, 100, 70, 80, 90],
            "company": ["DL", "AF", "AF", "DL", "DL", "TR"],
        })
    >>> join_agg = AggJoiner(
            tables=[
                (aux, "from_airport", ["total_passengers", "company"]),
            ],
            main_keys="airportId",
            operations=["mean", "mode"],
        )
    >>> join_agg.fit_transform(main)
        airportId airportName company_mode  total_passengers_mean
    0          1   Paris CDG           AF             103.333333
    1          2      NY JFK           DL              80.000000
    """  # noqa: E501

    def __init__(
        self,
        tables: tuple[DataFrameLike | str, ArrayLike | str, ArrayLike | str],
        main_keys: str | list[str],
        operations: str | list[str] | None = None,
        suffixes: str | list[str] | None = None,
    ):
        self.tables = tables
        self.main_keys = main_keys
        self.operations = operations
        self.suffixes = suffixes

    def fit(
        self,
        X: DataFrameLike,
        y: ArrayLike | SeriesLike | None = None,
    ) -> "AggJoiner":
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
        skrub_px, _ = get_df_namespace(*[table for (table, _, _) in self.tables_])

        if self.operations is None:
            operations = ["mean", "mode"]
        else:
            operations = np.atleast_1d(self.operations).tolist()
        self.operations_ = operations

        num_operations, categ_operations = split_num_categ_operations(operations)

        self.agg_tables_ = []
        for (table, cols_to_join, cols_to_agg), suffix in zip(
            self.tables_, self.suffixes_
        ):
            agg_table = skrub_px.aggregate(
                table,
                cols_to_join,
                cols_to_agg,
                num_operations,
                categ_operations,
                suffix,
            )
            agg_table = self._screen(agg_table, y)
            self.agg_tables_.append((agg_table, cols_to_join))

        return self

    def transform(self, X: DataFrameLike) -> DataFrameLike:
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

        check_is_fitted(self, "agg_tables_")
        skrub_px, _ = get_df_namespace(*[table for (table, _, _) in self.tables_])

        for aux, aux_keys in self.agg_tables_:
            X = skrub_px.join(
                left=X,
                right=aux,
                left_on=self.main_keys_,
                right_on=aux_keys,
            )

        return X

    def _screen(
        self,
        agg_table: DataFrameLike,
        y: DataFrameLike | SeriesLike | ArrayLike,
    ) -> DataFrameLike:
        """Only keep aggregated features which correlation with
        y is above some threshold.
        """
        # TODO: Add logic
        return agg_table

    def check_input(self, X: DataFrameLike) -> None:
        """Perform a check on column names data type and suffixes.

        Parameters
        ----------
        X : DataFrameLike
            The raw input to check.
        """
        if not hasattr(X, "__dataframe__"):
            raise TypeError(f"X must be a dataframe, got {type(X)}.")

        # Check main_keys
        self.main_keys_ = check_missing_columns(X, self.main_keys)

        # Check the number of elements in each tuple
        tables_ = []
        for idx, table_tuple in enumerate(self.tables):
            if not isinstance(table_tuple, tuple):
                raise TypeError(
                    f"'tables' must be a list of tuple, got {type(table_tuple)} "
                    f"at index {idx}."
                )

            table = table_tuple[0]
            if isinstance(table, str):
                if table != "X":
                    raise ValueError(
                        "If the dataframe is declared with a string, "
                        f"it can only be 'X', got '{table}'."
                    )
                table = deepcopy(X)
            elif not hasattr(table, "__dataframe__"):
                raise TypeError(
                    "'tables' must be a list of tuple and the first element of each"
                    f" tuple must be a dataFrame, got {type(table_tuple[0])} at index"
                    f" {idx}."
                )

            if len(table_tuple) == 2:
                # 'cols_to_agg' is missing, we define it as
                # the residuals between all columns of X and the cols to join.
                _, cols_to_join = table_tuple
                cols_to_agg = table.columns.drop(cols_to_join, errors="ignore")
            elif len(table_tuple) == 3:
                _, cols_to_join, cols_to_agg = table_tuple
            else:
                raise ValueError(
                    "Each tuple of 'tables' must have 2 or 3 elements, "
                    f"got {len(table_tuple)} for tuple at index {idx}."
                )

            cols_to_join = np.atleast_1d(cols_to_join).tolist()
            cols_to_agg = np.atleast_1d(cols_to_agg).tolist()

            if len(cols_to_join) != len(self.main_keys_):
                raise ValueError(
                    "The number of keys to join must match, got "
                    f"{self.main_keys_!r} for the base table and "
                    f"{cols_to_join!r} for the auxiliary table at index {idx}."
                )

            # check all columns to join and to aggregate belong to X columns
            table_cols = set(table.columns)
            input_cols = set([*cols_to_join, *cols_to_agg])
            missing_cols = input_cols - table_cols
            if len(missing_cols) > 0:
                raise ValueError(f"{missing_cols} are missing in table {idx}")

            tables_.append((table, cols_to_join, cols_to_agg))

        self.tables_ = tables_

        # Check suffixes
        self.suffixes_ = check_suffixes(self.suffixes, len(self.tables_))

        return


class AggTarget(BaseEstimator, TransformerMixin):
    """Aggregates the target ``y`` before joining its aggregation on the base dataframe.

    Uses pd.DataFrame or pl.DataFrame inputs.

    Parameters
    ----------
    main_keys : str or array-like
        Select the columns from the main table to use as keys during
        the aggregation of the target and during the join operation.

        If main_key refer to a single column, a single aggregation
        for this key will be generated and a single join will be performed.

        Otherwise, if main_key is a list of keys, the target will be
        aggregated using each key separately, then each aggregation of
        the target will be joined on the main table.

    operations : str or list of str, default=None
        Aggregation operations to perform on the auxiliary table.

        numerical : {"sum", "mean", "std", "min", "max", "hist(3)", "value_counts"}
            'hist' and 'value_counts' accepts an integer argument to parametrize
            the binning.

        categorical : {"mode", "count", "value_counts"}

        If set to None (the default), ['mean', 'mode'] will be used.

    suffixes : list of str, default=None
        The suffixes that will be add to each table columns in case of
        duplicate column names, similar to `("_x", "_y")` in :obj:`~pandas.merge`.

        If set to None, we will use the table index, by looking at their
        order in 'tables', e.g. for a duplicate columns: price (main table),
        price_1 (auxiliary table 1), price_2 (auxiliary table 2), etc.

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
    >>> X = pd.DataFrame({
            "flightId": range(1, 7),
            "from_airport": [1, 1, 1, 2, 2, 2],
            "total_passengers": [90, 120, 100, 70, 80, 90],
            "company": ["DL", "AF", "AF", "DL", "DL", "TR"],
        })
    >>> y = np.array([1, 1, 0, 0, 1, 1])
    >>> join_agg = AggTarget(
            main_keys="company",
            operations=["mean", "max"],
        )
    >>> join_agg.fit_transform(X, y)
        flightId  from_airport  total_passengers company  y0_max   y0_mean
    0         1             1                90      DL       1  0.666667
    1         2             1               120      AF       1  0.500000
    2         3             1               100      AF       1  0.500000
    3         4             2                70      DL       1  0.666667
    4         5             2                80      DL       1  0.666667
    5         6             2                90      TR       1  1.000000
    """

    def __init__(
        self,
        main_keys: str | list[str],
        operations: str | list[str] | None = None,
        suffixes: str | list[str] | None = None,
    ):
        self.main_keys = main_keys
        self.operations = operations
        self.suffixes = suffixes

    def fit(
        self,
        X: DataFrameLike,
        y: DataFrameLike | SeriesLike | ArrayLike,
    ) -> "AggTarget":
        """Aggregate the target ``y`` based on keys from ``X``.

        Parameters
        ----------
        X : dataframe
            Must contains the columns names defined in ``main_keys``.
        y : dataframe or array-like
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

        y_[self.main_keys_] = X[self.main_keys_]

        num_operations, categ_operations = split_num_categ_operations(self.operations_)

        self.all_agg_y_ = []
        for main_key, suffix in zip(self.main_keys_, self.suffixes_):
            agg_y = skrub_px.aggregate(
                y_,
                cols_to_join=main_key,
                cols_to_agg=self.cols_to_agg_,
                num_operations=num_operations,
                categ_operations=categ_operations,
                suffix=suffix,
            )
            self.all_agg_y_.append((agg_y, main_key))

        return self

    def transform(self, X: DataFrameLike) -> DataFrameLike:
        """Left-join pre-aggregated tables on `X`.

        Parameters
        ----------
        X : dataframe,
            The input data to transform.

        Returns
        -------
        X_transformed : dataframe,
            The augmented input.
        """
        check_is_fitted(self, "all_agg_y_")
        agg_px, _ = get_df_namespace(X)

        for agg_y, main_key in self.all_agg_y_:
            X = agg_px.join(
                left=X,
                right=agg_y,
                left_on=main_key,
                right_on=main_key,
            )

        return X

    def check_input(
        self,
        X: DataFrameLike,
        y: DataFrameLike | SeriesLike | ArrayLike,
    ) -> DataFrameLike:
        """Perform a check on column names data type and suffixes.

        Parameters
        ----------
        X : dataframe
            The raw input to check.
        y : array-like or dataframe,
            The raw target to check.

        Returns
        -------
        y_ : dataframe,
            Transformation of the target.
        """
        _, px = get_df_namespace(X)

        main_keys = check_missing_columns(X, self.main_keys)

        # If y is not a dataframe, we convert it.
        if hasattr(y, "__dataframe__"):
            y_ = y.copy()
        else:
            y_ = np.atleast_2d(y.copy())
            y_ = px.DataFrame(y_)

            # If y is Series or an array derived from a
            # Series, we need to transpose it.
            if len(y_) == 1 and len(y) != 1:
                y_ = y_.T

            if hasattr(y, "name"):
                # y is a Series
                cols = [y.name]
            else:
                cols = ["y" + col for col in np.asarray(y_.columns, dtype=str)]
            y_.columns = cols

        # Check lengths
        if y_.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y length must match, got {X.shape[0]} and {y_.shape[0]}."
            )

        self.cols_to_agg_ = y_.columns
        self.main_keys_ = main_keys

        self.suffixes_ = check_suffixes(self.suffixes, n_aux_tables=1)

        if self.operations is None:
            y_type = type_of_target(y_)
            if y_type in ["continuous", "continuous-multioutput"]:
                operations = ["mean"]
            else:
                operations = ["mode"]
        else:
            operations = np.atleast_1d(self.operations).tolist()
        self.operations_ = operations

        return y_
