import re

import pandas as pd
import polars as pl
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from skrub._agg_joiner import AggJoiner, AggTarget, get_namespace, split_num_categ_ops

main = pd.DataFrame(
    {
        "userId": [1, 1, 1, 2, 2, 2],
        "movieId": [1, 3, 6, 318, 6, 1704],
        "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
        "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
    }
)


def test_agg_join():
    agg_join = AggJoiner(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
            (main, ["movieId"], ["rating"]),
        ],
        main_key=["userId", "movieId"],
        suffixes=["_user", "_movie"],
        agg_ops=["mean", "mode"],
    )
    main_user_movie = agg_join.fit_transform(main)

    expected = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
            "genre_mode_user": ["drama", "drama", "drama", "sf", "sf", "sf"],
            "rating_mean_user": [4.0, 4.0, 4.0, 3.0, 3.0, 3.0],
            "rating_mean_movie": [4.0, 4.0, 3.0, 3.0, 3.0, 4.0],
        }
    )
    assert_frame_equal(main_user_movie, expected)

    agg_join = AggJoiner(
        tables=[
            ("X", ["userId"], ["rating", "genre"]),
            ("X", ["movieId"], ["rating"]),
        ],
        main_key=["userId", "movieId"],
        suffixes=["_user", "_movie"],
        agg_ops=["mean", "mode"],
    )
    X_string_user_movie = agg_join.fit_transform(main)

    assert_frame_equal(main_user_movie, X_string_user_movie)


def test_agg_join_check_input():
    msg = "X must be a dataframe, got <class 'numpy.ndarray'>."
    agg_join = AggJoiner(
        tables=[
            ("X", "userId"),
        ],
        main_key="userId",
    )
    with pytest.raises(TypeError, match=msg):
        agg_join.check_input(main.values)

    # check main key missing
    agg_join = AggJoiner(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
        ],
        main_key=["wrong_key"],
    )
    msg = (
        "Got main_key=['wrong_key'], but column not in "
        "X.columns ['userId', 'movieId', 'rating', 'genre']."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        agg_join.check_input(main)

    # check too many main keys
    agg_join = AggJoiner(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
        ],
        main_key=["userId", "movieId"],
    )
    msg = "The number of main keys must be either 1 or match the number of tables"
    with pytest.raises(ValueError, match=msg):
        agg_join.check_input(main)

    # check main key length
    agg_join = AggJoiner(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
            (main, ["userId"], ["rating", "genre"]),
        ],
        main_key="userId",
    )
    agg_join.check_input(main)
    assert agg_join.main_keys_ == ["userId", "userId"]

    # check missing agg or keys cols in tables
    agg_join = AggJoiner(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
            (main, ["movieId"], ["wrong_key"]),
        ],
        main_key=["userId", "movieId"],
    )
    msg = "{'wrong_key'} are missing in table 2"
    with pytest.raises(ValueError, match=msg):
        agg_join.check_input(main)

    # check no suffix with one table
    agg_join = AggJoiner(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
        ],
        main_key=["userId"],
    )
    agg_join.check_input(main)
    assert agg_join.suffixes_ == [""]

    # check no suffix with multiple tables
    agg_join = AggJoiner(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
            (main, ["movieId"], ["rating"]),
        ],
        main_key=["userId"],
    )
    agg_join.check_input(main)
    assert agg_join.suffixes_ == ["_1", "_2"]

    # check incorrect suffix type
    agg_join = AggJoiner(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
            (main, ["movieId"], ["rating"]),
        ],
        main_key=["userId"],
        suffixes=1,
    )
    msg = "Suffixes must be a list of string matching the number of tables, got: '1'"
    with pytest.raises(ValueError, match=msg):
        agg_join.check_input(main)

    # check inconsistent number of suffixes
    agg_join = AggJoiner(
        tables=[
            (main, "userId", ["rating", "genre"]),
            (main, "movieId", ["rating"]),
        ],
        main_key=["userId"],
        suffixes=["_user", "_movie", "_tag"],
    )
    msg = (
        "Suffixes must be None or match the number of tables"
        ", got: '['_user', '_movie', '_tag']'"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        agg_join.check_input(main)


def test_agg_join_default_ops():
    # check default agg_ops
    agg_join = AggJoiner(
        tables=[
            (main, "userId", ["rating", "genre"]),
        ],
        main_key=["userId"],
    )
    agg_join.fit(main)
    assert agg_join.agg_ops_ == ["mean", "mode"]

    # check invariant agg_ops input
    agg_join = AggJoiner(
        tables=[
            (main, "userId", ["rating", "genre"]),
        ],
        main_key=["userId"],
        agg_ops=["min", "max", "mode"],
    )
    agg_join.fit(main)
    assert agg_join.agg_ops_ == ["min", "max", "mode"]

    # check not supported agg_ops
    agg_join = AggJoiner(
        tables=[
            (main, "userId", ["rating", "genre"]),
        ],
        main_key=["userId"],
        agg_ops=["most_frequent", "mode"],
    )

    expected_ops = [
        "sum",
        "mean",
        "std",
        "min",
        "max",
        "hist",
        "value_counts",
        "mode",
        "count",
        "value_counts",
    ]
    msg = f"'agg_ops' options are {expected_ops}, got: 'most_frequent'."
    with pytest.raises(ValueError, match=re.escape(msg)):
        agg_join.fit(main)

    # check split ops
    num_ops, categ_ops = split_num_categ_ops(
        ["mean", "std", "min", "mode", "max", "sum"]
    )
    assert num_ops == ["mean", "std", "min", "max", "sum"]
    assert categ_ops == ["mode"]


def test_get_namespace():
    agg_px, _ = get_namespace([main, main])
    assert agg_px.__name__ == "skrub._agg_pandas"

    agg_px, _ = get_namespace([pl.DataFrame(main), pl.DataFrame(main)])
    assert agg_px.__name__ == "skrub._agg_polars"

    msg = "Only Pandas or Polars dataframes are currently supported"
    with pytest.raises(TypeError, match=msg):
        get_namespace([main, main.values])

    msg = "Mixing polars lazyframes and dataframes is not supported."
    print(pl.LazyFrame(main))
    with pytest.raises(TypeError, match=msg):
        get_namespace([pl.DataFrame(main), pl.LazyFrame(main)])


def test_tuples_tables():
    # check 'tables' is a list of tuple
    msg = (
        "'tables' must be a list of tuple, "
        "got <class 'pandas.core.frame.DataFrame'> at index 0."
    )
    agg_join = AggJoiner(
        tables=[
            main,
        ],
        main_key="userId",
    )
    with pytest.raises(TypeError, match=msg):
        agg_join.fit(main)

    # check 2d tuples are equivalent to 3d tuples
    agg_join = AggJoiner(
        tables=[
            (main, "userId"),
        ],
        main_key="userId",
    )
    agg_join.fit(main)

    _, _, cols_to_agg = agg_join.tables_[0]
    cols_to_agg_expected = ["movieId", "rating", "genre"]
    assert_array_equal(cols_to_agg, cols_to_agg_expected)

    # check bad 1d tuple
    msg = (
        "Each tuple of 'tables' must have 2 or 3 elements, got 4 for tuple at index 0."
    )
    agg_join = AggJoiner(
        tables=[
            (main, "userId", "rating", "hello"),
        ],
        main_key="userId",
    )
    with pytest.raises(ValueError, match=msg):
        agg_join.fit(main)

    # check non dataframe
    msg = (
        "'tables' must be a list of tuple and the first element of each "
        "tuple must be a DataFrame, got <class 'numpy.ndarray'> at index 0."
    )
    agg_join = AggJoiner(
        tables=[
            (main.values, "userId"),
        ],
        main_key="userId",
    )
    with pytest.raises(TypeError, match=msg):
        agg_join.fit(main)


def test_X_string_placeholder():
    msg = "If the dataframe is declared with a string, it can only be 'X', got 'Y'."
    agg_join = AggJoiner(
        tables=[
            ("Y", "userId"),
        ],
        main_key="userId",
    )
    with pytest.raises(ValueError, match=msg):
        agg_join.fit(main)


y = pd.DataFrame(dict(rating=[4.0, 4.0, 4.0, 3.0, 2.0, 4.0]))


@pytest.mark.parametrize(
    "y, col_name",
    [
        (y, "rating"),
        (y["rating"], "rating"),
        (y.values, "y0"),
        (y.values.tolist(), "y0"),
    ],
)
def test_agg_target(y, col_name):
    agg_target = AggTarget(
        main_key=["userId"],
        suffixes=["_user"],
        agg_ops=["hist(2)", "value_counts"],
    )
    main_transformed = agg_target.fit_transform(main, y)

    main_transformed_expected = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
            f"{col_name}_(1.999, 3.0]_user": [0, 0, 0, 2, 2, 2],
            f"{col_name}_(3.0, 4.0]_user": [3, 3, 3, 1, 1, 1],
            f"{col_name}_2.0_user": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            f"{col_name}_3.0_user": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            f"{col_name}_4.0_user": [3.0, 3.0, 3.0, 1.0, 1.0, 1.0],
        }
    )

    assert_frame_equal(main_transformed, main_transformed_expected)


def test_agg_t_missing_agg_ops():
    agg_target = AggTarget(
        main_key=["userId"],
        suffixes=["_user"],
    )

    # y is continuous
    y = pd.DataFrame(dict(rating=[4.0, 4.1, 4.2, 4.3, 4.4, 4.5]))
    agg_target.fit(main, y)
    assert agg_target.agg_ops_ == ["mean"]

    # y is categorical
    y = pd.DataFrame(dict(rating=["1", "2", "3", "1", "2", "3"]))
    agg_target.fit(main, y)
    assert agg_target.agg_ops_ == ["mode"]


def test_agg_t_check_input():
    agg_target = AggTarget(
        main_key=["userId"],
        suffixes=["_user"],
    )
    msg = (
        "Only Pandas or Polars dataframes are currently supported, "
        "got [<class 'numpy.ndarray'>]."
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        agg_target.fit(main.values, y)

    msg = "X and y length must match, got 6 and 2."
    with pytest.raises(ValueError, match=re.escape(msg)):
        agg_target.fit(main, y["rating"][:2])


def test_no_aggregation_exception():
    agg_target = AggTarget(
        main_key="userId",
        agg_ops=[],
    )
    msg = "No aggregation has been performed"
    with pytest.raises(ValueError, match=msg):
        agg_target.fit(main, y)


def test_wrong_args_ops():
    msg = "Operator 'mean' doesn't take any argument, got '2'"
    agg_target = AggTarget(
        main_key="userId",
        agg_ops=["mean(2)"],
    )
    with pytest.raises(ValueError, match=msg):
        agg_target.fit(main, y)
