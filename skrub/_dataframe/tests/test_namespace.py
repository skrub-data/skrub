import pandas as pd
import pytest

import skrub._dataframe._pandas as skrub_pd
import skrub._dataframe._polars as skrub_pl
from skrub._dataframe._namespace import get_df_namespace
from skrub._dataframe._polars import POLARS_SETUP

main = pd.DataFrame(
    {
        "userId": [1, 1, 1, 2, 2, 2],
        "movieId": [1, 3, 6, 318, 6, 1704],
        "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
        "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
    }
)


def test_get_namespace_pandas():
    skrub_px, px = get_df_namespace(main, main)
    assert skrub_px is skrub_pd
    assert px is pd

    with pytest.raises(TypeError, match=r"(?=.*Only Pandas or Polars)(?=.*supported)"):
        get_df_namespace(main, main.values)


@pytest.mark.skipif(not POLARS_SETUP, reason="Polars is not available")
def test_get_namespace_polars():
    import polars as pl

    skrub_px, px = get_df_namespace(pl.DataFrame(main), pl.DataFrame(main))
    assert skrub_px is skrub_pl
    assert px is pl

    with pytest.raises(TypeError, match=r"(?=.*Mixing Pandas)(?=.*Polars)"):
        get_df_namespace(main, pl.DataFrame(main))

    with pytest.raises(
        TypeError, match=r"(?=.*Mixing)(?=.*lazyframes)(?=.*dataframes)"
    ):
        get_df_namespace(pl.DataFrame(main), pl.LazyFrame(main))
