import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from skrub._dataframe._pandas import (
    aggregate,
    join,
    make_dataframe,
    make_series,
    rename_columns,
)

main = pd.DataFrame(
    {
        "userId": [1, 1, 1, 2, 2, 2],
        "movieId": [1, 3, 6, 318, 6, 1704],
        "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
        "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
    }
)


def test_join():
    joined = join(left=main, right=main, left_on="movieId", right_on="movieId")
    expected = main.merge(main, on="movieId", how="left")
    assert_frame_equal(joined, expected)


def test_simple_agg():
    aggregated = aggregate(
        table=main,
        key="movieId",
        cols_to_agg=["rating", "genre"],
        num_operations="mean",
        categ_operations="mode",
    )
    aggfunc = {
        "genre_mode": ("genre", pd.Series.mode),
        "rating_mean": ("rating", "mean"),
    }
    expected = main.groupby("movieId").agg(**aggfunc)
    assert_frame_equal(aggregated, expected)


def test_value_counts_agg():
    aggregated = aggregate(
        table=main,
        key="userId",
        cols_to_agg="rating",
        num_operations="value_counts",
        categ_operations=None,
        suffix="_user",
    )
    expected = pd.DataFrame(
        {
            "rating_2.0_user": [0.0, 1.0],
            "rating_3.0_user": [0.0, 1.0],
            "rating_4.0_user": [3.0, 1.0],
            "userId": [1, 2],
        }
    )
    assert_frame_equal(aggregated, expected)

    aggregated = aggregate(
        table=main,
        key="userId",
        cols_to_agg="rating",
        num_operations="hist(2)",
        categ_operations=None,
        suffix="_user",
    )
    expected = pd.DataFrame(
        {
            "rating_(1.999, 3.0]_user": [0, 2],
            "rating_(3.0, 4.0]_user": [3, 1],
            "userId": [1, 2],
        }
    )
    assert_frame_equal(aggregated, expected)


def test_incorrect_dataframe_inputs():
    with pytest.raises(TypeError, match=r"(?=.*pandas dataframes)(?=.*array)"):
        join(left=main.values, right=main, left_on="movieId", right_on="movieId")

    with pytest.raises(TypeError, match=r"(?=.*pandas dataframe)(?=.*array)"):
        aggregate(
            table=main.values,
            key="movieId",
            cols_to_agg="rating",
            num_operations="mean",
        )


def test_no_agg_operation():
    with pytest.raises(ValueError, match=r"(?=.*No aggregation)"):
        aggregate(
            table=main,
            key="movieId",
            cols_to_agg="rating",
            num_operations=None,
            categ_operations=None,
        )


@pytest.mark.parametrize("dtypes", [None, {"a": "int64", "b": "category"}])
def test_make_dataframe(dtypes):
    X = dict(a=[1, 2], b=["z", "e"])

    expected_df = pd.DataFrame(dict(a=[1, 2], b=["z", "e"]))
    if dtypes is not None:
        expected_df = expected_df.astype(dtypes)

    df = make_dataframe(X, index=[0, 1], dtypes=dtypes)
    assert_frame_equal(df, expected_df)


@pytest.mark.parametrize("dtype", [None, "category"])
def test_make_series(dtype):
    X = ["1", "2", "3"]
    expected_series = pd.Series(X, dtype=dtype)
    series = make_series(X, index=[0, 1, 2], dtype=dtype)
    assert_series_equal(series, expected_series)


def test_rename_columns():
    df = pd.DataFrame({"a column": [1], "another": [1]})
    df = rename_columns(df, str.swapcase)
    assert list(df.columns) == ["A COLUMN", "ANOTHER"]
