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
    # TODO
    pass


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

    # Check providing one col for each aux_table
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

    # This should work
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main], keys=["userId"], cols=["rating"], operations=["mean"]
    )
    multi_agg_joiner.fit(main)
    assert multi_agg_joiner._operations == [["mean"]]

    # This should not work
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys=["userId"],
        cols=[["rating"], ["rating"]],
        operations="mean",
    )
    with pytest.raises(ValueError):
        multi_agg_joiner.fit_transform(main)

    # This should work
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys=["userId"],
        cols=[["rating"], ["rating"]],
        operations=[["mean"], ["mean"]],
    )
    multi_agg_joiner.fit(main)
    assert multi_agg_joiner._operations == [["mean"], ["mean"]]

    # This should not work
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys="userId",
        cols=[["rating"], ["rating"]],
        operations=["mean", "mean", "mode"],
    )
    with pytest.raises(ValueError):
        multi_agg_joiner.fit_transform(main)

    # This should work
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys="userId",
        cols=[["rating"], ["rating"]],
        operations=[["mean"], ["mean", "mode"]],
    )
    multi_agg_joiner.fit(main)
    assert multi_agg_joiner._operations == [["mean"], ["mean", "mode"]]
    pass


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
