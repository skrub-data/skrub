import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.pipeline import make_pipeline

from skrub._dataframe._polars import POLARS_SETUP

if POLARS_SETUP:
    import polars as pl
    from polars.testing import assert_frame_equal as assert_frame_equal_pl

from skrub._agg_joiner import AggJoiner, AggTarget, split_num_categ_operations

main = pd.DataFrame(
    {
        "userId": [1, 1, 1, 2, 2, 2],
        "movieId": [1, 3, 6, 318, 6, 1704],
        "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
        "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
    }
)


ASSERT_TUPLES = [(main, pd, assert_frame_equal)]
if POLARS_SETUP:
    ASSERT_TUPLES.append((pl.DataFrame(main), pl, assert_frame_equal_pl))


@pytest.mark.parametrize("use_X_placeholder", [False, True])
@pytest.mark.parametrize(
    "X, px, assert_frame_equal_",
    ASSERT_TUPLES,
)
def test_simple_fit_transform(use_X_placeholder, X, px, assert_frame_equal_):
    aux = X if not use_X_placeholder else "X"

    agg_join_user = AggJoiner(
        aux_table=aux,
        aux_key="userId",
        main_key="userId",
        cols=["rating", "genre"],
        suffix="_user",
    )

    agg_join_movie = AggJoiner(
        aux_table=aux,
        aux_key="movieId",
        main_key="movieId",
        cols=["rating"],
        suffix="_movie",
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
def test_polars_unavailable_operation():
    agg_join = AggJoiner(
        aux_table="X",
        aux_key="movieId",
        cols="rating",
        main_key="userId",
        operation=["value_counts"],
    )
    with pytest.raises(ValueError, match=r"(?=.*value_counts)(?=.*supported)"):
        agg_join.fit(pl.DataFrame(main))


def test_input_single_table():
    agg_join = AggJoiner(
        aux_table=main,
        aux_key="userId",
        cols="genre",
        main_key="userId",
    )
    with pytest.raises(TypeError, match=r"(?=.*dataframe)(?=.*ndarray)"):
        agg_join.check_input(main.values)

    # check too many main keys
    agg_join = AggJoiner(
        aux_table=main,
        aux_key="userId",
        cols=["rating", "genre"],
        main_key=["userId", "movieId"],
    )
    with pytest.raises(ValueError, match=r"(?=.*keys to join must match)"):
        agg_join.check_input(main)

    # check too many foreign keys
    agg_join = AggJoiner(
        aux_table=main,
        aux_key=["userId", "movieId"],
        cols=["rating", "genre"],
        main_key="userId",
    )
    with pytest.raises(ValueError, match=r"(?=.*keys to join must match)"):
        agg_join.check_input(main)

    # check multiple keys, same length
    agg_join = AggJoiner(
        aux_table=main,
        aux_key=["userId", "movieId"],
        cols=["rating", "genre"],
        main_key=["userId", "movieId"],
    )
    agg_join.check_input(main)
    # aux_key_ is 2d since we iterate over it
    assert agg_join.aux_key_ == [["userId", "movieId"]]
    assert agg_join.main_key_ == ["userId", "movieId"]

    # check no suffix with one table
    agg_join = AggJoiner(
        aux_table=main,
        aux_key="userId",
        cols=["rating", "genre"],
        main_key="userId",
    )
    agg_join.check_input(main)
    assert agg_join.suffix_ == ["_1"]

    # check inconsistent number of suffixes
    agg_join = AggJoiner(
        aux_table=main,
        aux_key="movieId",
        cols="rating",
        main_key="userId",
        suffix=["_user", "_movie", "_tag"],
    )
    with pytest.raises(
        ValueError, match=r"(?=.*suffix)(?=.*match the number of tables)"
    ):
        agg_join.check_input(main)

    # check missing cols
    agg_join = AggJoiner(
        aux_table=main,
        aux_key=["movieId", "userId"],
        main_key=["movieId", "userId"],
    )
    agg_join.check_input(main)
    agg_join.cols_ == ["rating", "genre"]


def test_input_multiple_tables():
    # check foreign key are list of list
    agg_join = AggJoiner(
        aux_table=[main, main],
        aux_key=["userId", "userId"],
        cols=[["rating"], ["rating"]],
        main_key="userId",
    )
    error_msg = (
        r"(?=.*number of tables)(?=.*number of aux_key)" r"(?=.*For multiple tables)"
    )
    with pytest.raises(ValueError, match=error_msg):
        agg_join.fit_transform(main)

    agg_join = AggJoiner(
        aux_table=[main, main],
        aux_key=[["userId", "userId"]],
        cols=[["rating"], ["rating"]],
        main_key="userId",
    )
    with pytest.raises(ValueError, match=error_msg):
        agg_join.fit_transform(main)

    # check cols are list of list
    agg_join = AggJoiner(
        aux_table=[main, main],
        aux_key=[["userId"], ["userId"]],
        cols=["rating", "rating"],
        main_key="userId",
    )
    error_msg = (
        r"(?=.*number of tables)(?=.*number of cols)" r"(?=.*For multiple tables)"
    )
    with pytest.raises(ValueError, match=error_msg):
        agg_join.fit_transform(main)

    agg_join = AggJoiner(
        aux_table=[main, main],
        aux_key=[["userId"], ["userId"]],
        cols=[["rating", "rating"]],
        main_key="userId",
    )
    with pytest.raises(ValueError, match=error_msg):
        agg_join.fit_transform(main)

    # check suffixes with multiple tables
    agg_join = AggJoiner(
        aux_table=[main, main],
        aux_key=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
        main_key="userId",
    )
    agg_join.check_input(main)
    assert agg_join.suffix_ == ["_1", "_2"]


def test_wrong_key():
    # check main key missing
    agg_join = AggJoiner(
        aux_table=main,
        aux_key="userId",
        cols=["rating", "genre"],
        main_key="wrong_key",
    )
    match = r"(?=.*main_key)(?=.*X.column)"
    with pytest.raises(ValueError, match=match):
        agg_join.check_input(main)

    # check main key missing
    agg_join = AggJoiner(
        aux_table=main,
        aux_key="wrong_key",
        cols=["rating", "genre"],
        main_key="userId",
    )
    match = r"(?=.*aux_key)(?=.*table.column)"
    with pytest.raises(ValueError, match=match):
        agg_join.check_input(main)

    # check missing agg or keys cols in tables
    agg_join = AggJoiner(
        aux_table=main,
        aux_key="userId",
        cols="wrong_key",
        main_key="userId",
    )
    match = r"(?=.*cols)(?=.*not in)(?=.*table.columns)"
    with pytest.raises(ValueError, match=match):
        agg_join.check_input(main)


def test_agg_join_default_operations():
    # check default operations
    agg_join = AggJoiner(
        aux_table=main,
        aux_key="userId",
        cols=["rating", "genre"],
        main_key="userId",
    )
    agg_join.fit(main)
    assert agg_join.operation_ == ["mean", "mode"]

    # check invariant operations input
    agg_join = AggJoiner(
        aux_table=main,
        aux_key="userId",
        cols=["rating", "genre"],
        main_key="userId",
        operation=["min", "max", "mode"],
    )
    agg_join.fit(main)
    assert agg_join.operation_ == ["min", "max", "mode"]

    # check not supported operations
    agg_join = AggJoiner(
        aux_table=main,
        aux_key="userId",
        cols=["rating", "genre"],
        main_key="userId",
        operation=["most_frequent", "mode"],
    )
    match = r"(?=.*operations options are)"
    with pytest.raises(ValueError, match=match):
        agg_join.fit(main)

    # check split ops
    num_ops, categ_ops = split_num_categ_operations(
        ["mean", "std", "min", "mode", "max", "sum"]
    )
    assert num_ops == ["mean", "std", "min", "max", "sum"]
    assert categ_ops == ["mode"]


def test_X_wrong_string_placeholder():
    agg_join = AggJoiner(
        aux_table="Y",
        aux_key="userId",
        main_key="userId",
        cols="genre",
    )
    with pytest.raises(ValueError, match=r"(?=.*string)(?=.*'X')"):
        agg_join.fit(main)


y = pd.DataFrame(dict(rating=[4.0, 4.0, 4.0, 3.0, 2.0, 4.0]))


@pytest.mark.parametrize(
    "y, col_name",
    [
        (y, "rating"),
        (y["rating"], "rating"),
        (y.values, "y_0"),
        (y.values.tolist(), "y_0"),
    ],
)
def test_agg_target(y, col_name):
    agg_target = AggTarget(
        main_key="userId",
        suffix="_user",
        operation=["hist(2)", "value_counts"],
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
        main_key="userId",
        suffix="_user",
    )

    # y is continuous
    y = pd.DataFrame(dict(rating=[4.0, 4.1, 4.2, 4.3, 4.4, 4.5]))
    agg_target.fit(main, y)
    assert agg_target.operation_ == ["mean"]

    # y is categorical
    y = pd.DataFrame(dict(rating=["1", "2", "3", "1", "2", "3"]))
    agg_target.fit(main, y)
    assert agg_target.operation_ == ["mode"]


def test_agg_target_check_input():
    agg_target = AggTarget(
        main_key="userId",
        suffix="_user",
    )
    match = r"(?=.*X must be a dataframe)"
    with pytest.raises(TypeError, match=match):
        agg_target.fit(main.values, y)

    match = r"(?=.*length)(?=.*match)"
    with pytest.raises(ValueError, match=match):
        agg_target.fit(main, y["rating"][:2])


def test_no_aggregation_exception():
    agg_target = AggTarget(
        main_key="userId",
        operation=[],
    )
    with pytest.raises(ValueError, match=r"(?=.*No aggregation)"):
        agg_target.fit(main, y)


def test_wrong_args_ops():
    agg_target = AggTarget(
        main_key="userId",
        operation="mean(2)",
    )
    with pytest.raises(ValueError, match=r"(?=.*'mean')(?=.*argument)"):
        agg_target.fit(main, y)
