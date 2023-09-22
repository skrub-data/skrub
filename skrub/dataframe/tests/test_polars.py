import pandas as pd
import pytest

from skrub.dataframe import POLARS_SETUP
from skrub.dataframe._polars import aggregate, join

if POLARS_SETUP:
    import polars as pl
    from polars.testing import assert_frame_equal

    main = pl.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
        }
    )

POLARS_MISSING_MSG = "Polars is not available"


@pytest.mark.skipif(not POLARS_SETUP, reason=POLARS_MISSING_MSG)
def test_join():
    joined = join(left=main, right=main, left_on="movieId", right_on="movieId")
    expected = main.join(main, on="movieId", how="left")
    assert_frame_equal(joined, expected)


@pytest.mark.skipif(not POLARS_SETUP, reason=POLARS_MISSING_MSG)
def test_simple_agg():
    aggregated = aggregate(
        table=main,
        key="movieId",
        cols_to_agg="rating",
        num_operations="mean",
    )
    aggfunc = pl.col("rating").mean().alias("rating_mean")
    expected = main.group_by("movieId").agg(aggfunc)
    # As group_by parallizes threads, the row order of its output isn't
    # deterministic. Hence, we need to set check_row_order to False.
    assert_frame_equal(aggregated, expected, check_row_order=False)


@pytest.mark.skipif(not POLARS_SETUP, reason=POLARS_MISSING_MSG)
def test_mode_agg():
    aggregated = aggregate(
        table=main,
        key="movieId",
        cols_to_agg="genre",
        categ_operations=["mode"],
    )
    expected = pl.DataFrame(
        {
            "genre_mode": ["drama", "drama", "sf", "sf", "comedy"],
            "movieId": [3, 1, 318, 1704, 6],
        }
    )
    assert_frame_equal(aggregated, expected, check_row_order=False)


@pytest.mark.skipif(not POLARS_SETUP, reason=POLARS_MISSING_MSG)
def test_incorrect_dataframe_inputs():
    with pytest.raises(TypeError, match=r"(?=.*polars dataframes)(?=.*pandas)"):
        join(left=pd.DataFrame(main), right=main, left_on="movieId", right_on="movieId")

    with pytest.raises(TypeError, match=r"(?=.*polars dataframe)(?=.*pandas)"):
        aggregate(
            table=pd.DataFrame(main),
            key="movieId",
            cols_to_agg="rating",
            num_operations="mean",
        )
