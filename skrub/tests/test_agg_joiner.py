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

    agg_joiner_user = AggJoiner(
        aux_table=aux,
        aux_key="userId",
        main_key="userId",
        cols=["rating", "genre"],
        suffix="_user",
    )

    agg_joiner_movie = AggJoiner(
        aux_table=aux,
        aux_key="movieId",
        main_key="movieId",
        cols=["rating"],
        suffix="_movie",
    )

    agg_joiner = make_pipeline(agg_joiner_user, agg_joiner_movie)
    main_user_movie = agg_joiner.fit_transform(X)

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
    agg_joiner = AggJoiner(
        aux_table="X",
        aux_key="movieId",
        cols="rating",
        main_key="userId",
        operation=["value_counts"],
    )
    with pytest.raises(ValueError, match=r"(?=.*value_counts)(?=.*supported)"):
        agg_joiner.fit(pl.DataFrame(main))


def test_input_single_table():
    # check too many main keys
    agg_joiner = AggJoiner(
        aux_table=main,
        aux_key="userId",
        cols=["rating", "genre"],
        main_key=["userId", "movieId"],
    )
    with pytest.raises(
        ValueError, match=r"(?=.*Cannot join on different numbers of columns)"
    ):
        agg_joiner.fit(main)

    # check too many foreign keys
    agg_joiner = AggJoiner(
        aux_table=main,
        aux_key=["userId", "movieId"],
        cols=["rating", "genre"],
        main_key="userId",
    )
    with pytest.raises(
        ValueError, match=r"(?=.*Cannot join on different numbers of columns)"
    ):
        agg_joiner.fit(main)

    # check providing only key
    agg_joiner = AggJoiner(
        aux_table=main,
        key=["userId"],
        cols=["rating", "genre"],
    )
    agg_joiner.fit(main)

    # check providing multiple keys
    agg_joiner = AggJoiner(
        aux_table=main,
        key=["userId", "movieId"],
        cols=["rating", "genre"],
    )
    agg_joiner.fit(main)

    # check multiple keys, same length
    agg_joiner = AggJoiner(
        aux_table=main,
        aux_key=["userId", "movieId"],
        cols=["rating", "genre"],
        main_key=["userId", "movieId"],
    )
    agg_joiner.fit(main)
    # aux_key_ is 2d since we iterate over it
    assert agg_joiner._aux_key == ["userId", "movieId"]
    assert agg_joiner._main_key == ["userId", "movieId"]

    # check no suffix
    agg_joiner = AggJoiner(
        aux_table=main,
        aux_key="userId",
        cols=["rating", "genre"],
        main_key="userId",
    )
    agg_joiner.fit(main)
    assert agg_joiner.suffix == ""

    # check inconsistent number of suffixes
    agg_joiner = AggJoiner(
        aux_table=main,
        aux_key="movieId",
        cols="rating",
        main_key="userId",
        suffix=["_user", "_movie", "_tag"],
    )
    with pytest.raises(ValueError, match=r"(?='suffix' must be a string.*)"):
        agg_joiner.fit(main)

    # check missing cols
    agg_joiner = AggJoiner(
        aux_table=main,
        aux_key=["movieId", "userId"],
        main_key=["movieId", "userId"],
    )
    agg_joiner.fit(main)
    agg_joiner.cols == ["rating", "genre"]


def test_input_multiple_tables():
    # check foreign key are list of list
    agg_joiner = AggJoiner(
        aux_table=[main, main],
        aux_key=["userId", "userId"],
        cols=[["rating"], ["rating"]],
        main_key="userId",
    )
    error_msg = r"(?=.*must be a dataframe or 'X', got <class 'list'>)"
    with pytest.raises(TypeError, match=error_msg):
        agg_joiner.fit_transform(main)


def test_wrong_key():
    # check providing key and extra aux_key
    agg_joiner = AggJoiner(
        aux_table=main,
        aux_key=["userId"],
        key=["userId"],
        cols=["rating", "genre"],
    )
    with pytest.raises(ValueError, match=r"(?=.*not a combination of both.)"):
        agg_joiner.fit(main)

    # check providing key and extra main_key
    agg_joiner = AggJoiner(
        aux_table=main,
        main_key=["userId"],
        key=["userId"],
        cols=["rating", "genre"],
    )
    with pytest.raises(ValueError, match=r"(?=.*not a combination of both.)"):
        agg_joiner.fit(main)

    # check providing key and extra aux_key and main_key
    agg_joiner = AggJoiner(
        aux_table=main,
        aux_key=["userId"],
        main_key=["userId"],
        key=["userId"],
        cols=["rating", "genre"],
    )
    with pytest.raises(ValueError, match=r"(?=.*not a combination of both.)"):
        agg_joiner.fit(main)

    # check main key missing
    agg_joiner = AggJoiner(
        aux_table=main,
        aux_key="userId",
        cols=["rating", "genre"],
        main_key="wrong_key",
    )
    match = r"(?=.*columns cannot be used for joining because they do not exist)"
    with pytest.raises(ValueError, match=match):
        agg_joiner.fit(main)

    # check aux key missing
    agg_joiner = AggJoiner(
        aux_table=main,
        aux_key="wrong_key",
        cols=["rating", "genre"],
        main_key="userId",
    )
    match = r"(?=.*columns cannot be used for joining because they do not exist)"
    with pytest.raises(ValueError, match=match):
        agg_joiner.fit(main)

    # check missing agg or keys cols in tables
    agg_joiner = AggJoiner(
        aux_table=main,
        aux_key="userId",
        cols="wrong_key",
        main_key="userId",
    )
    match = "All 'cols' must be present in 'aux_table'"
    with pytest.raises(ValueError, match=match):
        agg_joiner.fit(main)


def test_agg_joiner_default_operations():
    # check default operations
    agg_joiner = AggJoiner(
        aux_table=main,
        aux_key="userId",
        cols=["rating", "genre"],
        main_key="userId",
    )
    agg_joiner.fit(main)
    assert agg_joiner.operation == ["mean", "mode"]

    # check invariant operations input
    agg_joiner = AggJoiner(
        aux_table=main,
        aux_key="userId",
        cols=["rating", "genre"],
        main_key="userId",
        operation=["min", "max", "mode"],
    )
    agg_joiner.fit(main)
    assert agg_joiner.operation == ["min", "max", "mode"]

    # check not supported operations
    agg_joiner = AggJoiner(
        aux_table=main,
        aux_key="userId",
        cols=["rating", "genre"],
        main_key="userId",
        operation=["most_frequent", "mode"],
    )
    match = r"(?=.*operations options are)"
    with pytest.raises(ValueError, match=match):
        agg_joiner.fit(main)

    # check split ops
    num_ops, categ_ops = split_num_categ_operations(
        ["mean", "std", "min", "mode", "max", "sum"]
    )
    assert num_ops == ["mean", "std", "min", "max", "sum"]
    assert categ_ops == ["mode"]


def test_X_wrong_string_placeholder():
    agg_joiner = AggJoiner(
        aux_table="Y",
        aux_key="userId",
        main_key="userId",
        cols="genre",
    )
    with pytest.raises(ValueError, match=r"(?=.*dataframe)(?=.*'X')"):
        agg_joiner.fit(main)


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
