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
def test_cols(main, px):
    main = px.DataFrame(main)

    # This should work
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=main,
        keys="userId",
        cols=["rating"],
    )
    multi_agg_joiner.fit(main)
    assert multi_agg_joiner._cols == ["rating"]

    # This should work
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys="userId",
        cols=[["rating"], ["rating"]],
    )
    multi_agg_joiner.fit(main)
    assert multi_agg_joiner._cols == [["rating"], ["rating"]]

    # This should not work
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=main,
        keys="userId",
        cols=[["rating"], ["rating"]],
    )
    # TODO: raise an error here


@pytest.mark.parametrize("px", MODULES)
def test_operations(main, px):
    main = px.DataFrame(main)

    # This should work
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=main, keys="userId", cols="rating", operations=["mean"]
    )
    multi_agg_joiner.fit(main)
    assert multi_agg_joiner._operations == ["mean"]

    # This should work
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys="userId",
        cols=[["rating"], ["rating"]],
        operations="mean",
    )
    multi_agg_joiner.fit(main)
    assert multi_agg_joiner._operations == ["mean", "mean"]

    # This should work
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys="userId",
        cols=[["rating"], ["rating"]],
        operations=["mean", "mean"],
    )
    multi_agg_joiner.fit(main)
    assert multi_agg_joiner._operations == ["mean", "mean"]

    # This should not work
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        keys="userId",
        cols=[["rating"], ["rating"]],
        operations=["mean", "mean", "mode"],
    )
    # TODO: raise an error here
    pass


def test_input_multiple_tables(main):
    # check foreign key are list of list
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        main_key="userId",
        aux_keys=["userId", "userId"],
        cols=[["rating"], ["rating"]],
    )
    error_msg = (
        r"(?=.*number of tables)(?=.*number of aux_key)" r"(?=.*For multiple tables)"
    )
    # with pytest.raises(ValueError, match=error_msg):
    #    multi_agg_joiner.fit_transform(main)

    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        main_key="userId",
        aux_keys=[["userId", "userId"]],
        cols=[["rating"], ["rating"]],
    )
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main)

    # check cols are list of list
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        main_key="userId",
        aux_keys=[["userId"], ["userId"]],
        cols=["rating", "rating"],
    )
    error_msg = (
        r"(?=.*number of tables)(?=.*number of cols)" r"(?=.*For multiple tables)"
    )
    # with pytest.raises(ValueError, match=error_msg):
    #    multi_agg_joiner.fit_transform(main)

    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        main_key="userId",
        aux_keys=[["userId"], ["userId"]],
        cols=[["rating", "rating"]],
    )
    # with pytest.raises(ValueError, match=error_msg):
    #    multi_agg_joiner.fit_transform(main)

    # check suffixes with multiple tables
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        main_key="userId",
        aux_keyss=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
    )
    multi_agg_joiner.check_input(main)
    assert multi_agg_joiner.suffix_ == ["_1", "_2"]
