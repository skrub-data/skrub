import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from skrub._dataframe._polars import POLARS_SETUP
from skrub._multi_agg_joiner import MultiAggJoiner


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
    main = px.DataFrame(main)
    aux = [main, main] if not use_X_placeholder else ["X", "X"]

    multi_agg_joiner = MultiAggJoiner(
        aux_tables=aux,
        main_keys=[["userId"], ["movieId"]],
        aux_keys=[["userId"], ["movieId"]],
        cols=[["rating", "genre"], ["rating"]],
        suffixes=["_user", "_movie"],
    )

    main_user_movie = multi_agg_joiner.fit_transform(main)

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


@pytest.mark.parametrize("px", MODULES)
def test_X_placeholder(main, px):
    main = px.DataFrame(main)

    multi_agg_joiner = MultiAggJoiner(
        aux_tables=["X", main],
        keys=[["userId"], ["userId"]],
    )
    multi_agg_joiner.fit_transform(main)

    multi_agg_joiner = MultiAggJoiner(
        aux_tables=["X", "X"],
        keys=[["userId"], ["userId"]],
    )
    multi_agg_joiner.fit_transform(main)

    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main, "X", main],
        keys=[["userId"], ["userId"], ["userId"], ["userId"]],
    )
    multi_agg_joiner.fit_transform(main)


@pytest.mark.parametrize("px", MODULES)
def test_check_dataframes(main, px):
    main = px.DataFrame(main)

    # Check aux_tables isn't an array
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=main,
        keys=["userId"],
    )
    with pytest.raises(
        ValueError, match=r"(?=`aux_tables` must be an iterable of dataframes or 'X'.)"
    ):
        multi_agg_joiner.fit_transform(main)

    # Check aux_tables is not a dataframe or "X"
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[1],
        keys=["userId"],
    )
    with pytest.raises(
        ValueError, match=r"(?=`aux_tables` must be an iterable of dataframes or 'X'.)"
    ):
        multi_agg_joiner.fit_transform(main)


@pytest.mark.parametrize("px", MODULES)
def test_keys(main, px):
    main = px.DataFrame(main)

    # Check only keys
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        keys=["userId"],
    )
    multi_agg_joiner.fit_transform(main)

    # Check multiple keys
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        keys=["userId", "movieId"],
    )
    multi_agg_joiner.fit_transform(main)

    # Check no keys at all
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
    )
    error_msg = r"Must pass EITHER `keys`, OR \(`main_keys` AND `aux_keys`\)."
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main)

    # Check multiple main_keys and aux_keys, same length
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        main_keys=["userId", "movieId"],
        aux_keys=["userId", "movieId"],
    )
    multi_agg_joiner.fit_transform(main)
    # aux_keys_ is 2d since we iterate over it
    assert multi_agg_joiner._main_keys == [["userId", "movieId"]]
    assert multi_agg_joiner._aux_keys == [["userId", "movieId"]]

    # Check too many main_keys
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        main_keys=["userId", "movieId"],
        aux_keys=["userId"],
    )
    with pytest.raises(
        ValueError, match=r"(?=.*Cannot join on different numbers of columns)"
    ):
        multi_agg_joiner.fit_transform(main)

    # Check too many aux_keys
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        main_keys=["userId"],
        aux_keys=["userId", "movieId"],
    )
    with pytest.raises(
        ValueError, match=r"(?=.*Cannot join on different numbers of columns)"
    ):
        multi_agg_joiner.fit_transform(main)

    # Check providing keys and extra main_keys
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        keys=["userId"],
        main_keys=["userId"],
    )
    with pytest.raises(ValueError, match=r"(?=.*not a combination of both.)"):
        multi_agg_joiner.fit_transform(main)

    # Check providing key and extra aux_keys
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        keys=["userId"],
        aux_keys=["userId"],
    )
    with pytest.raises(ValueError, match=r"(?=.*not a combination of both.)"):
        multi_agg_joiner.fit_transform(main)

    # Check main_keys doesn't exist in table
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        main_keys=["wrong_key"],
        aux_keys=["userId"],
    )
    error_msg = r"(?=.*columns cannot be used for joining because they do not exist)"
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main)

    # Check aux_keys doesn't exist in table
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        main_keys=["userId"],
        aux_keys=["wrong_key"],
    )
    error_msg = r"(?=.*columns cannot be used for joining because they do not exist)"
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main)

    # Check keys multiple tables
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys=[["userId"], ["userId"]],
    )
    multi_agg_joiner.fit_transform(main)

    # Check wrong keys lenght
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        main_keys=[["userId"], ["userId"]],
        aux_keys=[["userId"]],
    )
    error_msg = r"(?=`main_keys` and `aux_keys` have different lengths)"
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main)


@pytest.mark.parametrize("px", MODULES)
def test_cols(main, px):
    main = px.DataFrame(main)

    # Check providing one cols
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        keys=["userId"],
        cols=["rating"],
    )
    multi_agg_joiner.fit_transform(main)
    assert multi_agg_joiner._cols == [["rating"]]

    # Check providing one col for each aux_tables
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
    )
    multi_agg_joiner.fit_transform(main)
    assert multi_agg_joiner._cols == [["rating"], ["rating"]]

    # Check providing too many cols
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        keys=["userId"],
        cols=[["rating"], ["rating"]],
    )
    error_msg = r"The number of provided cols must match the number of"
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main)


@pytest.mark.parametrize("px", MODULES)
def test_operations(main, px):
    main = px.DataFrame(main)

    # Check one operation
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        keys=["userId"],
        cols=["rating"],
        operations=["mean"],
    )
    multi_agg_joiner.fit_transform(main)
    assert multi_agg_joiner._operations == [["mean"]]

    # Check list of operations
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
        operations="mean",
    )
    error_msg = r"Accepted inputs for operations are None, iterable of str"
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main)

    # Check one operation for each aux table
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
        operations=[["mean"], ["mean"]],
    )
    multi_agg_joiner.fit_transform(main)
    assert multi_agg_joiner._operations == [["mean"], ["mean"]]

    # Check badly formatted operation
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
        operations=["mean", "mean", "mode"],
    )
    error_msg = r"The number of provided operations must match the number of tables"
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main)

    # Check one and two operations
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
        operations=[["mean"], ["mean", "mode"]],
    )
    multi_agg_joiner.fit_transform(main)
    assert multi_agg_joiner._operations == [["mean"], ["mean", "mode"]]


@pytest.mark.parametrize("px", MODULES)
def test_suffixes(main, px):
    main = px.DataFrame(main)

    # check default suffixes with multiple tables
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
    )
    multi_agg_joiner.fit_transform(main)
    assert multi_agg_joiner._suffixes == ["_1", "_2"]

    # check suffixes when defined
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
        suffixes=["_this", "_works"],
    )
    multi_agg_joiner.fit_transform(main)
    assert multi_agg_joiner._suffixes == ["_this", "_works"]


@pytest.mark.parametrize("px", MODULES)
def test_tuple_parameters(main, px):
    main = px.DataFrame(main)
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=(main, main),
        keys=(("userId",), ("userId",)),
        cols=(("rating",), ("rating",)),
        operations=(("mean",), ("mean", "mode")),
        suffixes=("_1", "_2"),
    )
    multi_agg_joiner.fit_transform(main)


@pytest.mark.parametrize("px", MODULES)
def test_not_fitted_dataframe(main, px):
    main = px.DataFrame(main)
    not_main = px.DataFrame({"wrong": [1, 2, 3], "dataframe": [4, 5, 6]})

    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        keys=["userId"],
    )
    multi_agg_joiner.fit(main)
    error_msg = r"(?=.*columns cannot be used for joining because they do not exist)"
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.transform(not_main)
