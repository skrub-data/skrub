from abc import abstractmethod
from itertools import product

import numpy as np
import pandas as pd
try:
    import polars as pl
    POLARS_SETUP = True
except ImportError:
    POLARS_SETUP = False

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

NUM_OPS = ["mean", "std", "max", "min"]
CATEG_OPS = ["mode"]
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
        if op_name in NUM_OPS:
            num_ops.append(op_name)
        elif op_name in CATEG_OPS:
            categ_ops.append(op_name)
        else:
            ValueError(
                f"'ops' options are {ALL_OPS}, got: {op_name}."
            )
    return num_ops, categ_ops


class AssemblingEngine:
    """Helper class to perform the join and aggregate operations.

    This is an abstract base class that is specialized depending
    on the module of the dataframe used (Pandas or Polars).

    This helper class is in charge of mapping strings to aggregation
    operators of the respective module.
    """
    @classmethod
    def get_for(cls, tables):
        """Returns the AssemblingEngine implementation for given tables
        module.

        Parameters
        ----------
        tables : List of Tuple of (table, cols_to_join, cols_to_agg)
            We only use table to detect the module used.
        
        Returns
        -------
        assembling_engine : AssemblingEngine
            The suited AssemblingEngine implementation.
        """
        use_pandas = all(
            [isinstance(table, pd.DataFrame) for table, _, _ in tables]
        )
        use_polars = False
        if POLARS_SETUP:
            # we don't mix DataFrame and LazyFrame
            use_polars = (
                all(
                    [isinstance(table, pl.DataFrame) for table, _, _ in tables]
                )
                or all(
                    [isinstance(table, pl.LazyFrame) for table, _, _ in tables]
                )
            )
        if use_pandas:
            return PandasAssemblingEngine()
        elif use_polars:
            return PolarsAssemblingEngine()
        else:
            raise NotImplementedError(
                "Only Pandas or Polars DataFrame are currently supported."
            )

    @abstractmethod
    def agg(self, table, cols_to_join, cols_to_agg, ops):
        pass

    @abstractmethod
    def join(self, left, right, left_on, right_on):
        pass


def pandas_split_num_categ_cols(table):
    num_cols = table.select_dtypes("number").columns
    categ_cols = table.select_dtypes(
        ["object", "string", "category"]
    ).columns
    return num_cols, categ_cols
    

def pandas_get_agg_ops(cols, agg_ops):
    pandas_ops_mapping = {
        "mode": pd.Series.mode
    }
    stats = {}
    for col, op_name in product(cols, agg_ops):
        op = pandas_ops_mapping.get(op_name, op_name)
        stats[f"{col}_{op_name}"] = pd.NamedAgg(col, op)
    return stats


class PandasAssemblingEngine(AssemblingEngine):

    def agg(self, table, cols_to_join, cols_to_agg, agg_ops):
        
        num_cols, categ_cols = pandas_split_num_categ_cols(
            table[cols_to_agg]
        )
        num_ops, categ_ops = split_num_categ_ops(agg_ops)

        num_stats = pandas_get_agg_ops(num_cols, num_ops)
        categ_stats = pandas_get_agg_ops(categ_cols, categ_ops)
                    
        return table.groupby(cols_to_join).agg(**num_stats, **categ_stats)

    def join(self, left, right, left_on, right_on):
        
        return left.merge(
            right,
            how="left",
            left_on=left_on,
            right_on=right_on,
        )


def polars_split_num_categ_cols(table):
    
    num_cols = table.select(
        pl.col(pl.NUMERIC_DTYPES)
    ).columns
    
    categ_cols = table.select(
        pl.col(pl.Utf8)
    ).columns
    
    return num_cols, categ_cols


def polars_get_agg_ops(cols, agg_ops):
    stats, mode_cols = [], []
    for col, op_name in product(cols, agg_ops):
        op_dict = {
            "mean": pl.col(col).mean().alias(f"{col}_{op_name}"),
            "std": pl.col(col).std().alias(f"{col}_{op_name}"),
            "min": pl.col(col).min().alias(f"{col}_{op_name}"),
            "max": pl.col(col).max().alias(f"{col}_{op_name}"),
            "mode": pl.col(col).mode().alias(f"{col}_{op_name}"),
        }
        op = op_dict.get(op_name, None)
        if op is None:
            raise ValueError(
                f"Polars operation '{op}' is not supported. "
                f"Available: {list(op_dict)}"
            )
        stats.append(op)

        # mode() output needs a flattening post-processing
        if op_name == "mode":
            mode_cols.append(f"{col}_mode")
            
    return stats, mode_cols


class PolarsAssemblingEngine(AssemblingEngine):

    def agg(self, table, cols_to_join, cols_to_agg, agg_ops):

        num_cols, categ_cols = polars_split_num_categ_cols(
            table.select(cols_to_agg)
        )
        num_ops, categ_ops = split_num_categ_ops(agg_ops)

        num_ops, num_mode_cols = polars_get_agg_ops(num_cols, num_ops)
        categ_ops, categ_mode_cols = polars_get_agg_ops(categ_cols, categ_ops)

        all_ops = [*num_ops, *categ_ops]
        agg_table = table.groupby(cols_to_join).agg(all_ops)

        # flattening post-processing of mode() cols
        flatten_ops = []
        for col in [*num_mode_cols, *categ_mode_cols]:
            flatten_ops.append(
                pl.col(col).list[0].alias(col)
            )
        return agg_table.with_columns(flatten_ops)
          

    def join(self, left, right, left_on, right_on):
        
        return left.join(
            right,
            how="left",
            left_on=left_on,
            right_on=right_on,
        )
    

def check_cols(tables):
    """Check that all columns to join and columns to aggregate 
    belong to their respective dataframes.
    """
    for idx, (table, cols_to_join, cols_to_agg) in enumerate(tables, start=1):

        cols_to_join = np.atleast_1d(cols_to_join).tolist()
        cols_to_agg = np.atleast_1d(cols_to_agg).tolist()

        table_cols = set(table.columns)
        input_cols = set([*cols_to_join, *cols_to_agg])

        missing_cols = input_cols - table_cols
        if len(missing_cols) > 0:
            raise ValueError(f"{missing_cols} are missing in table {idx}")

    return
    

class JoinAggregator(BaseEstimator, TransformerMixin):
    """Perform aggregation on auxilliary dataframes before joining
    on the base dataframe.

    Apply numerical (mean, std, min, max) and categorical (mode) aggregation 
    operations on the columns to agg, selected by dtypes.
    
    The grouping columns used during the aggregation are the columns used 
    as keys for joining.

    These operations can run lazily by inputing polars LazyFrames. 
    Pandas and polars dataframes can't be mixed together, so the user has 
    to switch between format if needed.
    
    Parameters
    ----------
    tables : list of tuples
        List of (dataframe, columns_to_join, columns_to_agg) tuple
        specifying the auxilliary dataframes and their columns for joining 
        and aggregation operations.
    
        dataframe : {pandas.DataFrame, polars.DataFrame, polars.LazyFrame}
            The auxilliary data to aggregate and join.
        
        columns_to_join : str or array-like
            Select the columns from the dataframe to use as keys during the join operation.
        
        columns_to_agg : str or array-like
            Select the columns from the dataframe to use as values during 
            the aggregation operations.

    main_key : str or array-like
        Select the columns from the base table to use as keys during 
        the join operation.

    agg_ops : str or list of str, default=None
        Aggregation operations to perform on the auxilliary table.
        Options: {'mean', 'std', 'min', 'max', 'mode'}. If set to None, 
        ['mean', 'mode'] will be used.
    """

    def __init__(self, tables, main_key, agg_ops=None):
        self.tables = tables
        self.main_key = main_key
        self.agg_ops = agg_ops
        self.assembly_engine = AssemblingEngine.get_for(tables)

    def fit(self, X, y=None):
        """Fit the instance to the auxiliary tables by aggregating them
        and storing the outputs.

        Parameters
        ----------
        X : {pandas.Dataframe, polars.DataFrame, polars.LazyFrame}
            Input data, based table on which to left join the 
            auxilliary tables..
        
        y : array-like of shape (n_samples), default=None
            Used to compute the correlation between the generated covariates
            and the target for screening purposes.

        Returns
        -------
        :obj:`JoinAggregator`
            Fitted :class:`JoinAggregator` instance (self).
        """
        if self.main_key not in X.columns:
            raise ValueError(
                f"Got main_key={self.main_key!r}, but column not in {list(X.columns)}."
            )

        check_cols(self.tables)

        if self.agg_ops is None:
            agg_ops = ["mean", "mode"]
        else:
            agg_ops = np.atleast_1d(self.agg_ops).tolist()
        
        self.agg_tables_ = []
        for table, cols_to_join, cols_to_agg in self.tables:

            agg_table = self.assembly_engine.agg(
                table,
                cols_to_join,
                cols_to_agg,
                agg_ops,
            )
            agg_table = self._screen(agg_table, y)

            self.agg_tables_.append((agg_table, cols_to_join))
            
        return self

    def transform(self, X):
        """Transform `X` by left joining the pre-aggregated 
        auxiliary tables to it.

        Parameters
        ----------
        X : {pandas.DataFrame, polars.DataFrame, polars.LazyFrame}
            The input data to transform.
        """

        check_is_fitted(self, "agg_tables_")

        for aux, aux_key in self.agg_tables_:
            X = self.assembly_engine.join(
                left=X,
                right=aux,
                left_on=self.main_key,
                right_on=aux_key,
            )

        return X

    def _screen(self, agg_table, y):
        # TODO: Add logic
        return agg_table
