import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from skrub._agg_joiner import AggJoiner, AggTarget, split_num_categ_operations
from skrub._dataframe._polars import POLARS_SETUP


@pytest.fixture
def main():
    df = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
        }
    )
    return df


MODULES = [pd]
ASSERT_TUPLES = [(pd, assert_frame_equal)]

if POLARS_SETUP:
    import polars as pl
    from polars.testing import assert_frame_equal as assert_frame_equal_pl

    MODULES.append(pl)
    ASSERT_TUPLES.append((pl, assert_frame_equal_pl))


@pytest.mark.parametrize("use_X_placeholder", [False, True])
@pytest.mark.parametrize(
    "px, assert_frame_equal_",
    ASSERT_TUPLES,
)
def test_simple_fit_transform(main, use_X_placeholder, px, assert_frame_equal_):
    X = px.DataFrame(main)
    aux = X if not use_X_placeholder else "X"

    agg_joiner_user = AggJoiner(
        aux_table=aux,
        aux_key="userId",
        main_key="userId",
        cols=["rating", "genre"],
        suffix="_user",
    )

    main_user = agg_joiner_user.fit_transform(X)

    expected_user = px.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
            "genre_mode_user": ["drama", "drama", "drama", "sf", "sf", "sf"],
            "rating_mean_user": [4.0, 4.0, 4.0, 3.0, 3.0, 3.0],
        }
    )
    assert_frame_equal_(main_user, expected_user)

    agg_joiner_movie = AggJoiner(
        aux_table=aux,
        aux_key="movieId",
        main_key="movieId",
        cols=["rating"],
        suffix="_movie",
    )

    main_movie = agg_joiner_movie.fit_transform(X)

    expected_movie = px.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
            "rating_mean_movie": [4.0, 4.0, 3.0, 3.0, 3.0, 4.0],
        }
    )

    assert_frame_equal_(main_movie, expected_movie)


@pytest.mark.skipif(not POLARS_SETUP, reason="Polars is not available")
def test_polars_unavailable_operation(main):
    agg_joiner = AggJoiner(
        aux_table="X",
        main_key="userId",
        aux_key="movieId",
        cols="rating",
        operation=["value_counts"],
    )
    with pytest.raises(ValueError, match=r"(?=.*value_counts)(?=.*supported)"):
        agg_joiner.fit(pl.DataFrame(main))


@pytest.mark.parametrize("px", MODULES)
def test_keys(main, px):
    main = px.DataFrame(main)

    # Check only key
    agg_joiner = AggJoiner(
        aux_table=main,
        key="userId",
        cols=["rating", "genre"],
    )
    agg_joiner.fit(main)

    # Check multiple key
    agg_joiner = AggJoiner(
        aux_table=main,
        key=["userId", "movieId"],
        cols=["rating", "genre"],
    )
    agg_joiner.fit(main)

    # Check multiple main_key and aux_key, same length
    agg_joiner = AggJoiner(
        aux_table=main,
        main_key=["userId", "movieId"],
        aux_key=["userId", "movieId"],
        cols=["rating", "genre"],
    )
    agg_joiner.fit(main)
    assert agg_joiner._main_key == ["userId", "movieId"]
    assert agg_joiner._aux_key == ["userId", "movieId"]

    # Check too many main_key
    agg_joiner = AggJoiner(
        aux_table=main,
        main_key=["userId", "movieId"],
        aux_key="userId",
        cols=["rating", "genre"],
    )
    with pytest.raises(
        ValueError, match=r"(?=.*Cannot join on different numbers of columns)"
    ):
        agg_joiner.fit(main)

    # Check too many aux_key
    agg_joiner = AggJoiner(
        aux_table=main,
        main_key="userId",
        aux_key=["userId", "movieId"],
        cols=["rating", "genre"],
    )
    with pytest.raises(
        ValueError, match=r"(?=.*Cannot join on different numbers of columns)"
    ):
        agg_joiner.fit(main)

    # Check providing key and extra main_key
    agg_joiner = AggJoiner(
        aux_table=main,
        key="userId",
        main_key="userId",
        cols=["rating", "genre"],
    )
    with pytest.raises(ValueError, match=r"(?=.*not a combination of both.)"):
        agg_joiner.fit(main)

    # Check providing key and extra aux_key
    agg_joiner = AggJoiner(
        aux_table=main,
        key="userId",
        aux_key="userId",
        cols=["rating", "genre"],
    )
    with pytest.raises(ValueError, match=r"(?=.*not a combination of both.)"):
        agg_joiner.fit(main)

    # Check main_key doesn't exist in table
    agg_joiner = AggJoiner(
        aux_table=main,
        main_key="wrong_key",
        aux_key="userId",
        cols=["rating", "genre"],
    )
    match = r"(?=.*columns cannot be used because they do not exist)"
    with pytest.raises(ValueError, match=match):
        agg_joiner.fit(main)

    # Check aux_key doesn't exist in table
    agg_joiner = AggJoiner(
        aux_table=main,
        main_key="userId",
        aux_key="wrong_key",
        cols=["rating", "genre"],
    )
    match = r"(?=.*columns cannot be used because they do not exist)"
    with pytest.raises(ValueError, match=match):
        agg_joiner.fit(main)


@pytest.mark.parametrize("px", MODULES)
def test_suffix(main, px):
    main = px.DataFrame(main)

    # Check no suffix
    agg_joiner = AggJoiner(
        aux_table=main,
        key="userId",
        cols=["rating", "genre"],
    )
    agg_joiner.fit(main)
    assert agg_joiner.suffix == ""

    # Check inconsistent number of suffixes
    agg_joiner = AggJoiner(
        aux_table=main,
        key="userId",
        cols="rating",
        suffix=["_user", "_movie", "_tag"],
    )
    with pytest.raises(ValueError, match=r"(?='suffix' must be a string.*)"):
        agg_joiner.fit(main)


@pytest.mark.parametrize("px", MODULES)
def test_cols(main, px):
    main = px.DataFrame(main)

    # Check no cols
    agg_joiner = AggJoiner(
        aux_table=main,
        key=["movieId", "userId"],
    )
    agg_joiner.fit(main)
    agg_joiner._cols == ["rating", "genre"]

    # Check missing agg or keys cols in tables
    agg_joiner = AggJoiner(
        aux_table=main,
        key="userId",
        cols="wrong_key",
    )
    match = r"(?=.*columns cannot be used because they do not exist)"
    with pytest.raises(ValueError, match=match):
        agg_joiner.fit(main)


@pytest.mark.parametrize("px", MODULES)
def test_input_multiple_tables(main, px):
    main = px.DataFrame(main)

    # Check too many aux_table
    agg_joiner = AggJoiner(
        aux_table=[main, main],
        main_key="userId",
        aux_key=["userId", "userId"],
        cols=[["rating"], ["rating"]],
    )
    error_msg = r"(?=.*must be a dataframe or 'X', got <class 'list'>)"
    with pytest.raises(TypeError, match=error_msg):
        agg_joiner.fit_transform(main)


@pytest.mark.parametrize("px", MODULES)
def test_agg_joiner_default_operations(main, px):
    main = px.DataFrame(main)

    # Check default operations
    agg_joiner = AggJoiner(
        aux_table=main,
        key="userId",
        cols=["rating", "genre"],
    )
    agg_joiner.fit(main)
    assert agg_joiner._operation == ["mean", "mode"]

    # Check invariant operations input
    agg_joiner = AggJoiner(
        aux_table=main,
        key="userId",
        cols=["rating", "genre"],
        operation=["min", "max", "mode"],
    )
    agg_joiner.fit(main)
    assert agg_joiner._operation == ["min", "max", "mode"]

    # Check not supported operations
    agg_joiner = AggJoiner(
        aux_table=main,
        key="userId",
        cols=["rating", "genre"],
        operation=["most_frequent", "mode"],
    )
    match = r"(?=.*operations options are)"
    with pytest.raises(ValueError, match=match):
        agg_joiner.fit(main)

    # Check split ops
    num_ops, categ_ops = split_num_categ_operations(
        ["mean", "std", "min", "mode", "max", "sum"]
    )
    assert num_ops == ["mean", "std", "min", "max", "sum"]
    assert categ_ops == ["mode"]


@pytest.mark.parametrize("px", MODULES)
def test_X_wrong_string_placeholder(main, px):
    main = px.DataFrame(main)

    agg_joiner = AggJoiner(
        aux_table="Y",
        key="userId",
        cols="genre",
    )
    with pytest.raises(ValueError, match=r"(?=.*dataframe)(?=.*'X')"):
        agg_joiner.fit(main)


@pytest.mark.parametrize("px", MODULES)
def test_not_fitted_dataframe(main, px):
    main = px.DataFrame(main)
    not_main = px.DataFrame({"wrong": [1, 2, 3], "dataframe": [4, 5, 6]})

    agg_joiner = AggJoiner(
        aux_table=main,
        key="userId",
    )
    agg_joiner.fit(main)
    error_msg = r"(?=.*columns cannot be used because they do not exist)"
    with pytest.raises(ValueError, match=error_msg):
        agg_joiner.transform(not_main)


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
def test_agg_target(main, y, col_name):
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


def test_agg_target_missing_operations(main):
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


def test_agg_target_check_input(main):
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


def test_no_aggregation_exception(main):
    agg_target = AggTarget(
        main_key="userId",
        operation=[],
    )
    with pytest.raises(ValueError, match=r"(?=.*No aggregation)"):
        agg_target.fit(main, y)


def test_wrong_args_ops(main):
    agg_target = AggTarget(
        main_key="userId",
        operation="mean(2)",
    )
    with pytest.raises(ValueError, match=r"(?=.*'mean')(?=.*argument)"):
        agg_target.fit(main, y)
