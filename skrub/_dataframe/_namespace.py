import sys

import pandas as pd

import skrub._dataframe._pandas as skrub_pd
import skrub._dataframe._polars as skrub_pl


def is_pandas(dataframe):
    """Check whether the input is a Pandas dataframe.

    Parameters
    ----------
    dataframe : DataFrameLike
        The input dataframe

    Returns
    -------
    is_pandas : bool
        Whether the dataframe is a Pandas dataframe or not.
    """
    return isinstance(dataframe, pd.DataFrame)


def is_polars(dataframe):
    """Check whether the input is a Polars dataframe or lazyframe.

    Parameters
    ----------
    dataframe : DataFrameLike
        The input dataframe

    Returns
    -------
    is_polars : bool
        Whether the dataframe is a Polars dataframe/lazyframe or not.
    """
    if "polars" not in sys.modules:
        return False

    import polars as pl

    return isinstance(dataframe, (pl.DataFrame, pl.LazyFrame))


def get_df_namespace(*dfs):
    """Get the namespaces of dataframes.

    Introspects dataframes and returns their skrub namespace object
    ``skrub.dataframe._{pandas, polars}`` and the dataframe module
    ``{polars, pandas}`` itself.

    The dataframes passed in input need to come from the same module, otherwise a
    ``TypeError`` will be raised.

    The outputs of this function are denoted ``skrub_px`` and ``px`` in reference to
    the array API, returning namespace (NumPy, PyTorch and CuPy) as ``nx``.
    Since we deal with Polars (``pl``) and Pandas (``pd``), we use ``px``
    as a variable name.

    Parameters
    ----------
    dfs : DataFrameLike | list[DataFrameLike],
        The dataframes to extract modules from.

    Returns
    -------
    skrub_px : ModuleType
        Skrub namespace shared by dataframe objects.

    px : ModuleType
        Dataframe namespace, i.e. Pandas or Polars module.
    """
    # FIXME Pandas and Polars series will raise errors.
    if all([is_pandas(df) for df in dfs]):
        return skrub_pd, pd

    elif all([is_polars(df) for df in dfs]):
        import polars as pl

        if all([isinstance(df, pl.DataFrame) for df in dfs]) or all(
            [isinstance(df, pl.LazyFrame) for df in dfs]
        ):
            return skrub_pl, pl
        else:
            raise TypeError("Mixing Polars lazyframes and dataframes is not supported.")

    else:
        modules = [type(df).__module__ for df in dfs]
        if all([is_polars(df) or is_pandas(df) for df in dfs]):
            raise TypeError(
                "Mixing Pandas and Polars dataframes is not supported, "
                f"got {modules=!r}."
            )
        else:
            raise TypeError(
                "Only Pandas or Polars dataframes are currently supported, "
                f"got {modules=!r}."
            )
