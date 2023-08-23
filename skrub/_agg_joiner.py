from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

import skrub._agg_pandas as agg_pd
import skrub._agg_polars as agg_pl

NUM_OPS = ["sum", "mean", "std", "min", "max", "hist", "value_counts"]
CATEG_OPS = ["mode", "count", "value_counts"]
ALL_OPS = NUM_OPS + CATEG_OPS


def split_num_categ_ops(agg_ops):
    """Separate aggregagor operators input
    by their type.

    Parameters
    ----------
    agg_ops : list of str,
        The input operators names.

    Returns
    -------
    num_ops, categ_ops : Tuple of List of str
        List of operator names
    """
    num_ops, categ_ops = [], []
    for op_name in agg_ops:
        # hist(5) -> hist
        op_root = op_name.split("(")[0]
        if op_root in NUM_OPS:
            num_ops.append(op_name)
        if op_root in CATEG_OPS:
            categ_ops.append(op_name)
        if op_root not in ALL_OPS:
            raise ValueError(f"'agg_ops' options are {ALL_OPS}, got: '{op_name}'.")
    return num_ops, categ_ops


def get_namespace(dataframes):
    """Get namespace of dataframes

    Introspect `dataframes` arguments and return their skrub namespace object
    (pandas or polars based).

    Parameters
    ----------
    dataframes : dataframe objects

    Returns
    -------
    namespace : module
        Namespace shared by dataframe objects.
    """

    def _extract_module(table):
        return table.__class__.__module__.split(".")[0]

    def _is_module(dataframes, module):
        return all([_extract_module(table) == module for table in dataframes])

    if _is_module(dataframes, "pandas"):
        return agg_pd, pd

    elif _is_module(dataframes, "polars"):
        import polars as pl

        if all([isinstance(table, pl.DataFrame) for table in dataframes]) or all(
            [isinstance(table, pl.LazyFrame) for table in dataframes]
        ):
            return agg_pl, pl
        else:
            # XXX: polars series will raise this error
            raise TypeError("Mixing polars lazyframes and dataframes is not supported.")

    else:
        raise TypeError(
            "Only Pandas or Polars dataframes are currently supported, "
            f"got {[type(df) for df in dataframes]}."
        )


def check_missing_columns(X, main_key):
    main_keys = np.atleast_1d(main_key).tolist()
    missing_cols = set(main_keys) - set(X.columns)
    if len(missing_cols) > 0:
        raise ValueError(
            f"Got {main_key=!r}, but column not in {X.columns=}."
        )
    return main_keys


def check_suffixes(suffixes, main_keys):
    if suffixes is None:
        if len(main_keys) == 1:
            suffixes = [""]
        else:
            suffixes = [f"_{idx+1}" for idx in range(len(main_keys))]
    elif hasattr(suffixes, "__len__"):
        if len(suffixes) != len(main_keys):
            raise ValueError(
                "Suffixes must be None or match the "
                f"number of tables, got: {suffixes!r}"
            )
        suffixes = list(suffixes)
    else:
        raise ValueError(
            "Suffixes must be a list of string matching "
            f"the number of tables, got: '{suffixes}'"
        )
    return suffixes


class AggJoiner(BaseEstimator, TransformerMixin):
    """Aggregates auxiliary dataframes before joining them on the base dataframe.

    Apply numerical and categorical aggregation operations on the columns
    to aggregate, selected by dtypes. See the list of supported operations
    at the parameter `agg_ops`.

    The grouping columns used during the aggregation are the columns used
    as keys for joining.

    Uses :obj:`~pandas.DataFrame` inputs only.

    Parameters
    ----------
    tables : list of tuples
        List of (dataframe, columns_to_join, columns_to_agg) tuple
        specifying the auxiliary dataframes and their columns for joining
        and aggregation operations.

        dataframe : :obj:`~pandas.DataFrame` or str
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

    main_key : str or array-like
        Select the columns from the main table to use as keys during
        the join operation.

        If main_key refer to a single column, it will be used to join all tables.

        Otherwise, if main_key is a list it must specify the key for each table
        to join.

    agg_ops : str or list of str, optional
        Aggregation operations to perform on the auxiliary table.

        numerical : {"sum", "mean", "std", "min", "max", "hist", "value_counts"}

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
    :class:`FeatureAugmenter` :
        Augments a main table by automatically joining multiple
        auxiliary tables on it.

    Examples
    --------
    >>> main = pd.DataFrame({
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 1, 3, 6],
            "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
        })
    >>> join_agg = AggJoiner(
            tables=[
                (main, "userId", ["rating", "genre"]),
                (main, "movieId", ["rating"]),
            ],
            main_key=["userId", "movieId"],
            suffixes=["_user", "_movie"],
            agg_ops=["mean", "mode"],
        )
    >>> join_agg.fit_transform(main)
        userId  movieId  rating   genre genre_mode_user  rating_mean_user  rating_mean_movie
    0       1        1     4.0   drama           drama               4.0   3.5
    1       1        3     4.0   drama           drama               4.0   3.0
    2       1        6     4.0  comedy           drama               4.0   4.0
    3       2        1     3.0      sf              sf               3.0   3.5
    4       2        3     2.0  comedy              sf               3.0   3.0
    5       2        6     4.0      sf              sf               3.0   4.0
    """  # noqa: E501

    def __init__(self, tables, main_key, agg_ops=None, suffixes=None):
        self.tables = tables
        self.main_key = main_key
        self.agg_ops = agg_ops
        self.suffixes = suffixes

    def fit(self, X, y=None) -> "AggJoiner":
        """Aggregate auxiliary tables based on the main keys.

        Parameters
        ----------
        X : {pandas.Dataframe}
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
        agg_px, _ = get_namespace([table for (table, _, _) in self.tables_])

        if self.agg_ops is None:
            agg_ops = ["mean", "mode"]
        else:
            agg_ops = np.atleast_1d(self.agg_ops).tolist()
        self.agg_ops_ = agg_ops

        num_ops, categ_ops = split_num_categ_ops(agg_ops)

        self.agg_tables_ = []
        for (table, cols_to_join, cols_to_agg), suffix in zip(
            self.tables_, self.suffixes_
        ):
            agg_table = agg_px.aggregate(
                table,
                cols_to_join,
                cols_to_agg,
                num_ops,
                categ_ops,
                suffix,
            )
            agg_table = self._screen(agg_table, y)
            self.agg_tables_.append((agg_table, cols_to_join))

        return self

    def transform(self, X):
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

        check_is_fitted(self, "agg_tables_")
        agg_px, _ = get_namespace([table for (table, _, _) in self.tables_])

        for main_key, (aux, aux_key) in zip(self.main_keys_, self.agg_tables_):
            X = agg_px.join(
                left=X,
                right=aux,
                left_on=main_key,
                right_on=aux_key,
            )

        return X

    def _screen(self, agg_table, y):
        """Only keep aggregated features which correlation with
        y is above some threshold.
        """
        # TODO: Add logic
        return agg_table

    def check_input(self, X):
        """Perform a check on column names data type and suffixes.

        Parameters
        ----------
        X : dataframe
            The raw input to check.
        """
        if not hasattr(X, "__dataframe__"):
            raise TypeError(f"X must be a dataframe, got {type(X)}.")

        # Check main_keys
        # XXX: Do we want to enable main_keys with multiple columns?
        # e.g. [["user_id", "movie_id"], ["user_id"]]
        main_keys = check_missing_columns(X, self.main_key)

        n_main_keys, n_tables = len(main_keys), len(self.tables)
        if (n_main_keys != 1) and (n_main_keys != n_tables):
            raise ValueError(
                f"The number of main keys ({n_main_keys}) must be either 1 "
                f"or match the number of tables ({n_table}). "
            )

        # Ensure n_main_keys == n_tables
        if n_main_keys == 1:
            main_keys = main_keys * n_tables

        self.main_keys_ = main_keys

        # Check the number of elements in each tuple
        tables_ = []
        for idx, t_tuple in enumerate(self.tables):
            if not isinstance(t_tuple, tuple):
                raise TypeError(
                    f"'tables' must be a list of tuple, got {type(t_tuple)} "
                    f"at index {idx}."
                )

            table = t_tuple[0]
            if isinstance(table, str):
                if table != "X":
                    raise ValueError(
                        "If the dataframe is declared with a string, "
                        f"it can only be 'X', got '{table}'."
                    )
                table = deepcopy(X)
            elif not hasattr(table, "__dataframe__"):
                raise TypeError(
                    "'tables' must be a list of tuple and the first element of each "
                    f"tuple must be a DataFrame, got {type(t_tuple[0])} at index {idx}."
                )

            if len(t_tuple) == 2:
                # 'cols_to_agg' is missing, we define it as
                # the residuals between all columns of X and the cols to join.
                _, cols_to_join = t_tuple
                cols_to_agg = table.columns.drop(cols_to_join, errors="ignore")
                # XXX: alternatively, we could use the dataframe API
                # df_protocol = X.__dataframe__()
                # feature_names = np.asarray(
                #   list(df_protocol.column_names()), dtype=object
                # )
            elif len(t_tuple) == 3:
                _, cols_to_join, cols_to_agg = t_tuple
            else:
                raise ValueError(
                    "Each tuple of 'tables' must have 2 or 3 elements, "
                    f"got {len(t_tuple)} for tuple at index {idx}."
                )

            cols_to_join = np.atleast_1d(cols_to_join).tolist()
            cols_to_agg = np.atleast_1d(cols_to_agg).tolist()

            # check missing columns in table
            table_cols = set(table.columns)
            input_cols = set([*cols_to_join, *cols_to_agg])
            missing_cols = input_cols - table_cols
            if len(missing_cols) > 0:
                raise ValueError(f"{missing_cols} are missing in table {idx+1}")

            tables_.append((table, cols_to_join, cols_to_agg))

        self.tables_ = tables_

        # Check suffixes
        self.suffixes_ = check_suffixes(self.suffixes, main_keys)

        return


class AggTarget(BaseEstimator, TransformerMixin):
    """Aggregates the target `y` before joining its aggregation on the base dataframe.

    Uses :obj:`~pandas.DataFrame` inputs only.

    Parameters
    ----------
    main_key : str or array-like
        Select the columns from the main table to use as keys during
        the aggregation of the target and during the join operation.

        If main_key refer to a single column, a single aggregation
        for this key will be generated and a single join will be performed.

        Otherwise, if main_key is a list of keys, the target will be
        aggregated using each key separately, then each aggregation of
        the target will be joined on the main table.

    agg_ops : str or list of str, default=None
        Aggregation operations to perform on the auxiliary table.

        numerical : {"sum", "mean", "std", "min", "max", "hist", "value_counts"}

        categorical : {"mode", "count", "value_counts"}

        If set to None, ['mean', 'mode'] will be used.

    suffixes : list of str, default=None
        The suffixes that will be add to each table columns in case of
        duplicate column names, similar to `("_x", "_y")` in :obj:`~pandas.merge`.

        If set to None, we will use the table index, by looking at their
        order in 'tables', e.g. for a duplicate columns: price (main table),
        price_1 (auxiliary table 1), price_2 (auxiliary table 2), etc.

    See Also
    --------
    :class:`AggJoiner` :
        Aggregates auxiliary dataframes before joining them
        on the base dataframe.
    :class:`FeatureAugmenter` :
        Augments a main table by automatically joining multiple
        auxiliary tables on it.

    Examples
    --------
    >>> X = pd.DataFrame({
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 1, 3, 6],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
        })
    >>> y = pd.DataFrame(dict(rating=[4.0, 4.0, 4.0, 3.0, 2.0, 4.0]))
    >>> agg_target = AggTarget(
            main_key=["userId"],
            suffixes=["_user"],
            agg_ops=["value_counts"],
        )
    >>> agg_target.fit_transform(X, y)
        userId  movieId   genre  rating_2.0_user  rating_3.0_user  rating_4.0_user
    0       1        1   drama              0.0              0.0              3.0
    1       1        3   drama              0.0              0.0              3.0
    2       1        6  comedy              0.0              0.0              3.0
    3       2        1      sf              1.0              1.0              1.0
    4       2        3  comedy              1.0              1.0              1.0
    5       2        6      sf              1.0              1.0              1.0
    """

    def __init__(self, main_key, agg_ops=None, suffixes=None):
        self.main_key = main_key
        self.agg_ops = agg_ops
        self.suffixes = suffixes

    def fit(self, X, y):
        """Aggregate the target ``y`` based on keys from ``X``.

        Parameters
        ----------
        X : dataframe
            Must contains the columns names defined in ``main_key``.
        y : dataframe or array-like
            Must have the length of ``X``, with matching indices.
            The target can be continuous or discrete, with multiple columns.

            If the target is continuous, only numerical operations,
            listed in NUM_OPS, are applied.

            If the target is discrete, only categorical operations,
            listed in CATEG_OPS, are applied.

            Note that the target type is determined by
            :func:`sklearn.utils.multiclass.type_of_target`.

        Returns
        -------
        AggTarget
            Fitted AggTarget instance (self).
        """

        y_ = self.check_input(X, y)
        agg_px, _ = get_namespace([X, y_])

        y_[self.main_keys_] = X[self.main_keys_]

        num_ops, categ_ops = split_num_categ_ops(self.agg_ops_)

        self.all_agg_y_ = []
        for main_key, suffix in zip(self.main_keys_, self.suffixes_):
            agg_y = agg_px.aggregate(
                y_,
                cols_to_join=main_key,
                cols_to_agg=self.cols_to_agg_,
                num_ops=num_ops,
                categ_ops=categ_ops,
                suffix=suffix,
            )
            self.all_agg_y_.append((agg_y, main_key))

        return self

    def transform(self, X):
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
        agg_px, _ = get_namespace([X])

        for agg_y, main_key in self.all_agg_y_:
            X = agg_px.join(
                left=X,
                right=agg_y,
                left_on=main_key,
                right_on=main_key,
            )

        return X

    def check_input(self, X, y):
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
        _, px = get_namespace([X])

        main_keys = check_missing_columns(X, self.main_key)

        # If y is not a dataframe, we convert it.
        if hasattr(y, "__dataframe__"):
            y_ = y.copy()
        else:
            y_ = np.atleast_2d(y.copy())
            y_ = px.DataFrame(y_)

            # If y is pd.Series or an array derived from a
            # pd.Series, we need to transpose it.
            if len(y_) == 1 and len(y) != 1:
                y_ = y_.T

            # If y is a Series
            if hasattr(y, "name"):
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

        self.suffixes_ = check_suffixes(self.suffixes, main_keys)

        if self.agg_ops is None:
            y_type = type_of_target(y_)
            if y_type in ["continuous", "continuous-multioutput"]:
                agg_ops = ["mean"]
            else:
                agg_ops = ["mode"]
        else:
            agg_ops = np.atleast_1d(self.agg_ops).tolist()
        self.agg_ops_ = agg_ops

        return y_
