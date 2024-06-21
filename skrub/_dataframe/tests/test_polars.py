import pandas as pd
import pytest

from skrub._dataframe._polars import (
    aggregate,
    rename_columns,
)
from skrub.conftest import _POLARS_INSTALLED

if _POLARS_INSTALLED:
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
else:
    POLARS_MISSING_MSG = "Polars is not available"
    pytest.skip(reason=POLARS_MISSING_MSG, allow_module_level=True)


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


def test_incorrect_dataframe_inputs():
    with pytest.raises(TypeError, match=r"(?=.*polars dataframe)(?=.*pandas)"):
        aggregate(
            table=pd.DataFrame(main),
            key="movieId",
            cols_to_agg="rating",
            num_operations="mean",
        )


def test_rename_columns():
    df = pl.DataFrame({"a column": [1], "another": [1]})
    df = rename_columns(df, str.swapcase)
    assert list(df.columns) == ["A COLUMN", "ANOTHER"]
