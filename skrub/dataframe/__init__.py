from skrub.dataframe._namespace import get_df_namespace, is_pandas, is_polars
from skrub.dataframe._pandas import aggregate as pd_aggregate
from skrub.dataframe._pandas import join as pd_join
from skrub.dataframe._polars import aggregate as pl_aggregate
from skrub.dataframe._polars import join as pl_join
from skrub.dataframe._types import POLARS_SETUP, DataFrameLike, SeriesLike

__all__ = [
    POLARS_SETUP,
    DataFrameLike,
    SeriesLike,
    get_df_namespace,
    is_pandas,
    is_polars,
    pd_join,
    pd_aggregate,
    pl_join,
    pl_aggregate,
]
