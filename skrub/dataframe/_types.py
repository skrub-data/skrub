import pandas as pd

try:
    import polars as pl

    POLARS_SETUP = True
except ImportError:
    POLARS_SETUP = False

DataFrameLike = pd.DataFrame
SeriesLike = pd.Series
if POLARS_SETUP:
    DataFrameLike |= pl.DataFrame | pl.LazyFrame
    SeriesLike |= pl.Series
