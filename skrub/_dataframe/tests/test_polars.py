import pandas as pd
import pytest

from skrub._dataframe._polars import (
    POLARS_SETUP,
    aggregate,
    join,
    make_dataframe,
    make_series,
    rename_columns,
)

if POLARS_SETUP:
    import polars as pl
    from polars.testing import assert_frame_equal, assert_series_equal

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


def test_join():
    joined = join(left=main, right=main, left_on="movieId", right_on="movieId")
    expected = main.join(main, on="movieId", how="left")
    assert_frame_equal(joined, expected)


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
    with pytest.raises(TypeError, match=r"(?=.*polars dataframes)(?=.*pandas)"):
        join(left=pd.DataFrame(main), right=main, left_on="movieId", right_on="movieId")

    with pytest.raises(TypeError, match=r"(?=.*polars dataframe)(?=.*pandas)"):
        aggregate(
            table=pd.DataFrame(main),
            key="movieId",
            cols_to_agg="rating",
            num_operations="mean",
        )


@pytest.mark.parametrize("dtypes", [None, {"a": pl.Int64, "b": pl.Utf8}])
def test_make_dataframe(dtypes):
    X = dict(a=[1, 2], b=["z", "e"])

    expected_df = pl.DataFrame(dict(a=[1, 2], b=["z", "e"]))
    if dtypes is not None:
        expected_df = expected_df.cast(dtypes)

    df = make_dataframe(X, dtypes=dtypes)
    assert_frame_equal(df, expected_df)

    with pytest.raises(ValueError, match=r"(?=.*Polars dataframe)(?=.*index)"):
        make_dataframe(X, index=[0, 1])


@pytest.mark.parametrize("dtype", [None, pl.Int64])
def test_make_series(dtype):
    X = [1, 2, 3]
    expected_series = pl.Series(X, dtype=dtype)
    series = make_series(X, index=None, dtype=dtype)
    assert_series_equal(series, expected_series)

    with pytest.raises(ValueError, match=r"(?=.*Polars series)(?=.*index)"):
        make_series(X, index=[0, 1])


def test_rename_columns():
    df = pl.DataFrame({"a column": [1], "another": [1]})
    df = rename_columns(df, str.swapcase)
    assert list(df.columns) == ["A COLUMN", "ANOTHER"]
