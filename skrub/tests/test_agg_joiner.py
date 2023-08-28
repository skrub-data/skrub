import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.pipeline import make_pipeline

from skrub._utils import POLARS_SETUP

if POLARS_SETUP:
    import polars as pl
    from polars.testing import assert_frame_equal as assert_frame_equal_pl

from skrub.agg_joiner import _agg_pandas, _agg_polars
from skrub.agg_joiner._agg_joiner import (
    AggJoiner,
    AggTarget,
    get_df_namespace,
    split_num_categ_operations,
)

main = pd.DataFrame(
    {
        "userId": [1, 1, 1, 2, 2, 2],
        "movieId": [1, 3, 6, 318, 6, 1704],
        "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
        "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
    }
)


assert_tuples = [(main, pd, assert_frame_equal)]
if POLARS_SETUP:
    assert_tuples.append((pl.DataFrame(main), pl, assert_frame_equal_pl))


@pytest.mark.parametrize("use_X_placeholder", [False, True])
@pytest.mark.parametrize(
    "X, px, assert_frame_equal_",
    assert_tuples,
)
def test_agg_join(use_X_placeholder, X, px, assert_frame_equal_):
    aux = X if not use_X_placeholder else "X"

    agg_join_user = AggJoiner(
        tables=[
            (aux, "userId", ["rating", "genre"]),
        ],
        main_keys="userId",
        suffixes="_user",
        operations=["mean", "mode"],
    )
    agg_join_movie = AggJoiner(
        tables=[
            (aux, "movieId", ["rating"]),
        ],
        main_keys="movieId",
        suffixes="_movie",
        operations=["mean", "mode"],
    )
    agg_join = make_pipeline(agg_join_user, agg_join_movie)
    main_user_movie = agg_join.fit_transform(X)

    expected = px.DataFrame(
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
    assert_frame_equal_(main_user_movie, expected)


@pytest.mark.skipif(not POLARS_SETUP, reason="Polars is not available")
def test_polars_unavailable_ops():
    agg_join = AggJoiner(
        tables=[
            ("X", "movieId", ["rating"]),
        ],
        main_keys="userId",
        operations=["value_counts"],
    )
    with pytest.raises(ValueError, match=r"(?=.*value_counts)(?=.*supported)"):
        agg_join.fit(pl.DataFrame(main))


def test_agg_join_check_input():
    agg_join = AggJoiner(
        tables=[
            ("X", "userId"),
        ],
        main_keys="userId",
    )
    with pytest.raises(TypeError, match=r"(?=.*dataframe)(?=.*ndarray)"):
        agg_join.check_input(main.values)

    # check main key missing
    agg_join = AggJoiner(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
        ],
        main_keys=["wrong_key"],
    )
    with pytest.raises(ValueError, match=r"(?=.*main_keys)(?=.*column)"):
        agg_join.check_input(main)

    # check too many main keys
    agg_join = AggJoiner(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
        ],
        main_keys=["userId", "movieId"],
    )
    with pytest.raises(ValueError, match=r"(?=.*keys)(?=.*base)(?=.*auxiliary)"):
        agg_join.check_input(main)

    # check main key length
    agg_join = AggJoiner(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
            (main, ["userId"], ["rating", "genre"]),
        ],
        main_keys="userId",
    )
    agg_join.check_input(main)
    assert agg_join.main_keys_ == ["userId"]

    # check missing agg or keys cols in tables
    agg_join = AggJoiner(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
            (main, ["userId"], ["wrong_key"]),
        ],
        main_keys=["userId"],
    )
    with pytest.raises(ValueError, match=r"(?=.*missing)"):
        agg_join.check_input(main)

    # check no suffix with one table
    agg_join = AggJoiner(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
        ],
        main_keys=["userId"],
    )
    agg_join.check_input(main)
    assert agg_join.suffixes_ == [""]

    # check suffixes with multiple tables
    agg_join = AggJoiner(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
            (main, ["userId"], ["rating"]),
        ],
        main_keys=["userId"],
    )
    agg_join.check_input(main)
    assert agg_join.suffixes_ == ["_1", "_2"]

    # check incorrect suffix type
    agg_join = AggJoiner(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
            (main, ["movieId"], ["rating"]),
        ],
        main_keys=["userId"],
        suffixes=1,
    )
    with pytest.raises(ValueError, match=r"(?=.*list of string)"):
        agg_join.check_input(main)

    # check inconsistent number of suffixes
    agg_join = AggJoiner(
        tables=[
            (main, "userId", ["rating", "genre"]),
            (main, "movieId", ["rating"]),
        ],
        main_keys=["userId"],
        suffixes=["_user", "_movie", "_tag"],
    )

    with pytest.raises(ValueError, match=r"(?=.*Suffixes)(?=.*number)"):
        agg_join.check_input(main)


def test_agg_join_default_operations():
    # check default operations
    agg_join = AggJoiner(
        tables=[
            (main, "userId", ["rating", "genre"]),
        ],
        main_keys="userId",
    )
    agg_join.fit(main)
    assert agg_join.operations_ == ["mean", "mode"]

    # check invariant operations input
    agg_join = AggJoiner(
        tables=[
            (main, "userId", ["rating", "genre"]),
        ],
        main_keys="userId",
        operations=["min", "max", "mode"],
    )
    agg_join.fit(main)
    assert agg_join.operations_ == ["min", "max", "mode"]

    # check not supported operations
    agg_join = AggJoiner(
        tables=[
            (main, "userId", ["rating", "genre"]),
        ],
        main_keys=["userId"],
        operations=["most_frequent", "mode"],
    )

    with pytest.raises(ValueError, match=r"(?=.*operations options are)"):
        agg_join.fit(main)

    # check split ops
    num_ops, categ_ops = split_num_categ_operations(
        ["mean", "std", "min", "mode", "max", "sum"]
    )
    assert num_ops == ["mean", "std", "min", "max", "sum"]
    assert categ_ops == ["mode"]


def test_get_namespace():
    skrub_px, _ = get_df_namespace(main, main)
    assert skrub_px is _agg_pandas

    with pytest.raises(TypeError, match=r"(?=.*Pandas or Polars)(?=.*supported)"):
        get_df_namespace(main, main.values)

    if POLARS_SETUP:
        skrub_px, _ = get_df_namespace(pl.DataFrame(main), pl.DataFrame(main))
        assert skrub_px is _agg_polars

        with pytest.raises(TypeError, match=r"(?=.*Pandas)(?=.*Polars)"):
            get_df_namespace(main, pl.DataFrame(main))

        with pytest.raises(TypeError, match=r"(?=.*lazyframes)(?=.*dataframes)"):
            get_df_namespace(pl.DataFrame(main), pl.LazyFrame(main))


def test_tuples_tables():
    # check 'tables' is a list of tuple
    agg_join = AggJoiner(
        tables=[
            main,
        ],
        main_keys="userId",
    )
    with pytest.raises(TypeError, match=r"(?=.*list of tuple)(?=.*DataFrame)"):
        agg_join.fit(main)

    # check 2d tuples are equivalent to 3d tuples
    agg_join = AggJoiner(
        tables=[
            (main, "userId"),
        ],
        main_keys="userId",
    )
    agg_join.fit(main)

    _, _, cols_to_agg = agg_join.tables_[0]
    cols_to_agg_expected = ["movieId", "rating", "genre"]
    assert_array_equal(cols_to_agg, cols_to_agg_expected)

    # check bad 1d tuple
    agg_join = AggJoiner(
        tables=[
            (main, "userId", "rating", "hello"),
        ],
        main_keys="userId",
    )
    with pytest.raises(ValueError, match=r"(?=.*2 or 3 elements)"):
        agg_join.fit(main)

    # check non dataframe
    agg_join = AggJoiner(
        tables=[
            (main.values, "userId"),
        ],
        main_keys="userId",
    )
    with pytest.raises(TypeError, match=r"(?=.*dataFrame)(?=.*ndarray)"):
        agg_join.fit(main)


def test_X_string_placeholder():
    agg_join = AggJoiner(
        tables=[
            ("Y", "userId"),
        ],
        main_keys="userId",
    )
    with pytest.raises(ValueError, match=r"(?=.*string)(?=.*'X')"):
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
        main_keys=["userId"],
        suffixes=["_user"],
        operations=["hist(2)", "value_counts"],
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


def test_agg_target_missing_operations():
    agg_target = AggTarget(
        main_keys=["userId"],
        suffixes=["_user"],
    )

    # y is continuous
    y = pd.DataFrame(dict(rating=[4.0, 4.1, 4.2, 4.3, 4.4, 4.5]))
    agg_target.fit(main, y)
    assert agg_target.operations_ == ["mean"]

    # y is categorical
    y = pd.DataFrame(dict(rating=["1", "2", "3", "1", "2", "3"]))
    agg_target.fit(main, y)
    assert agg_target.operations_ == ["mode"]


def test_agg_target_check_input():
    agg_target = AggTarget(
        main_keys=["userId"],
        suffixes=["_user"],
    )
    with pytest.raises(TypeError, match=r"(?=.*Pandas)(?=.*Polars)"):
        agg_target.fit(main.values, y)

    with pytest.raises(ValueError, match=r"(?=.*length)(?=.*match)"):
        agg_target.fit(main, y["rating"][:2])


def test_no_aggregation_exception():
    agg_target = AggTarget(
        main_keys="userId",
        operations=[],
    )
    with pytest.raises(ValueError, match=r"(?=.*No aggregation)"):
        agg_target.fit(main, y)


def test_wrong_args_ops():
    agg_target = AggTarget(
        main_keys="userId",
        operations=["mean(2)"],
    )
    with pytest.raises(ValueError, match=r"(?=.*'mean')(?=.*argument)"):
        agg_target.fit(main, y)
