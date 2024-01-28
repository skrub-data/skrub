import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from skrub._dataframe._polars import POLARS_SETUP
from skrub._multi_joiner import MultiAggJoiner  # , MultiJoiner


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


@pytest.mark.parametrize("px", MODULES)
def test_keys(main, px):
    main = px.DataFrame(main)

    # Check only keys
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        keys=["userId"],
        cols=["rating", "genre"],
    )
    multi_agg_joiner.fit(main)

    # Check multiple keys
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        keys=["userId", "movieId"],
        cols=["rating", "genre"],
    )
    multi_agg_joiner.fit(main)

    # Check multiple main_key and aux_keys, same length
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        main_key=["userId", "movieId"],
        aux_keys=["userId", "movieId"],
        cols=["rating", "genre"],
    )
    multi_agg_joiner.fit(main)
    # aux_keys_ is 2d since we iterate over it
    assert multi_agg_joiner._main_key == ["userId", "movieId"]
    assert multi_agg_joiner._aux_keys == [["userId", "movieId"]]

    # Check too many main_key
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        main_key=["userId", "movieId"],
        aux_keys=["userId"],
        cols=["rating", "genre"],
    )
    with pytest.raises(
        ValueError, match=r"(?=.*Cannot join on different numbers of columns)"
    ):
        multi_agg_joiner.fit(main)

    # Check too many aux_keys
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        main_key="userId",
        aux_keys=["userId", "movieId"],
        cols=["rating", "genre"],
    )
    with pytest.raises(
        ValueError, match=r"(?=.*Cannot join on different numbers of columns)"
    ):
        multi_agg_joiner.fit(main)

    # Check providing keys and extra main_key
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        keys=["userId"],
        main_key="userId",
        cols=["rating", "genre"],
    )
    with pytest.raises(ValueError, match=r"(?=.*not a combination of both.)"):
        multi_agg_joiner.fit(main)

    # Check providing key and extra aux_keys
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        keys=["userId"],
        aux_keys=["userId"],
        cols=["rating", "genre"],
    )
    with pytest.raises(ValueError, match=r"(?=.*not a combination of both.)"):
        multi_agg_joiner.fit(main)

    # Check main_key doesn't exist in table
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        main_key="wrong_key",
        aux_keys=["userId"],
        cols=["rating", "genre"],
    )
    match = r"(?=.*columns cannot be used for joining because they do not exist)"
    with pytest.raises(ValueError, match=match):
        multi_agg_joiner.fit(main)

    # Check aux_keys doesn't exist in table
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        main_key="userId",
        aux_keys=["wrong_key"],
        cols=["rating", "genre"],
    )
    match = r"(?=.*columns cannot be used for joining because they do not exist)"
    with pytest.raises(ValueError, match=match):
        multi_agg_joiner.fit(main)


@pytest.mark.parametrize("px", MODULES)
def test_cols(main, px):
    main = px.DataFrame(main)

    # Check providing one col
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main],
        keys=["userId"],
        cols=["rating"],
    )
    multi_agg_joiner.fit(main)
    assert multi_agg_joiner._cols == [["rating"]]

    # Check providing one col for each aux_tables
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys=["userId"],
        cols=[["rating"], ["rating"]],
    )
    multi_agg_joiner.fit(main)
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
    multi_agg_joiner.fit(main)
    assert multi_agg_joiner._operations == [["mean"]]

    # Check providing a list
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys=["userId"],
        cols=[["rating"], ["rating"]],
        operations="mean",
    )
    error_msg = r"Accepted inputs for operations are None, iterable of str"
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main)

    # Check one operation for each aux_tables
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys=["userId"],
        cols=[["rating"], ["rating"]],
        operations=[["mean"], ["mean"]],
    )
    multi_agg_joiner.fit(main)
    assert multi_agg_joiner._operations == [["mean"], ["mean"]]

    # Check badly formatted operation
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys="userId",
        cols=[["rating"], ["rating"]],
        operations=["mean", "mean", "mode"],
    )
    error_msg = r"The number of provided operations must match the number of tables"
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main)

    # Check one and two operations
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys="userId",
        cols=[["rating"], ["rating"]],
        operations=[["mean"], ["mean", "mode"]],
    )
    multi_agg_joiner.fit(main)
    assert multi_agg_joiner._operations == [["mean"], ["mean", "mode"]]


# TODO: explode this test into multiple smaller ones
def test_input_multiple_tables(main):
    # check foreign key are list of list
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        main_key="userId",
        aux_keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
    )
    multi_agg_joiner.fit_transform(main)

    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        main_key="userId",
        aux_keys=[["userId", "userId"]],
        cols=[["rating"], ["rating"]],
    )
    with pytest.raises(ValueError):
        multi_agg_joiner.fit_transform(main)

    # check cols are list of list
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        main_key="userId",
        aux_keys=[["userId"], ["userId"]],
        cols=["rating", "rating"],
    )
    with pytest.raises(ValueError):
        multi_agg_joiner.fit_transform(main)

    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        main_key="userId",
        aux_keys=[["userId"], ["userId"]],
        cols=[["rating", "rating"]],
    )
    with pytest.raises(ValueError):
        multi_agg_joiner.fit_transform(main)


@pytest.mark.parametrize("px", MODULES)
def test_suffixes(main, px):
    main = px.DataFrame(main)

    # check default suffixes with multiple tables
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        main_key="userId",
        aux_keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
    )
    multi_agg_joiner.fit(main)
    assert multi_agg_joiner._suffixes == ["_1", "_2"]

    # check suffixes when defined
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        main_key="userId",
        aux_keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
        suffixes=["_this", "_works"],
    )
    multi_agg_joiner.fit(main)
    assert multi_agg_joiner._suffixes == ["_this", "_works"]
