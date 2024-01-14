import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from skrub._dataframe._polars import POLARS_SETUP

if POLARS_SETUP:
    import polars as pl
    from polars.testing import assert_frame_equal as assert_frame_equal_pl

from skrub._multi_joiner import MultiAggJoiner  # , MultiJoiner

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


def test_input_one_table():
    # TODO
    pass


def test_input_multiple_tables():
    # check foreign key are list of list
    agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        main_key="userId",
        aux_keys=["userId", "userId"],
        cols=[["rating"], ["rating"]],
    )
    error_msg = (
        r"(?=.*number of tables)(?=.*number of aux_key)" r"(?=.*For multiple tables)"
    )
    # with pytest.raises(ValueError, match=error_msg):
    #    agg_joiner.fit_transform(main)

    agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        main_key="userId",
        aux_keys=[["userId", "userId"]],
        cols=[["rating"], ["rating"]],
    )
    with pytest.raises(ValueError, match=error_msg):
        agg_joiner.fit_transform(main)

    # check cols are list of list
    agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        main_key="userId",
        aux_keys=[["userId"], ["userId"]],
        cols=["rating", "rating"],
    )
    error_msg = (
        r"(?=.*number of tables)(?=.*number of cols)" r"(?=.*For multiple tables)"
    )
    # with pytest.raises(ValueError, match=error_msg):
    #    agg_joiner.fit_transform(main)

    agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        main_key="userId",
        aux_keys=[["userId"], ["userId"]],
        cols=[["rating", "rating"]],
    )
    # with pytest.raises(ValueError, match=error_msg):
    #    agg_joiner.fit_transform(main)

    # check suffixes with multiple tables
    agg_joiner = MultiAggJoiner(
        aux_tables=[main, main],
        main_key="userId",
        aux_keyss=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
    )
    agg_joiner.check_input(main)
    assert agg_joiner.suffix_ == ["_1", "_2"]
