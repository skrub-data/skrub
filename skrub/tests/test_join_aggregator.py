import re

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from skrub._join_aggregator import (
    JoinAggregator,
    PandasAssemblingEngine,
    dispatch_assembling_engine,
    split_num_categ_ops,
)

main = pd.DataFrame(
    {
        "userId": [1, 1, 1, 2, 2, 2],
        "movieId": [1, 3, 6, 318, 6, 1704],
        "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
        "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
    }
)


def test_join_agg():
    join_agg = JoinAggregator(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
            (main, ["movieId"], ["rating"]),
        ],
        main_key=["userId", "movieId"],
        suffixes=["_user", "_movie"],
        agg_ops=["mean", "mode"],
    )
    main_user_movie = join_agg.fit_transform(main)

    expected = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
            "rating_mean_user": [4.0, 4.0, 4.0, 3.0, 3.0, 3.0],
            "genre_mode_user": ["drama", "drama", "drama", "sf", "sf", "sf"],
            "rating_mean_movie": [4.0, 4.0, 3.0, 3.0, 3.0, 4.0],
        }
    )
    assert_frame_equal(main_user_movie, expected)


def test_join_agg_check_cols():
    # check main key missing
    join_agg = JoinAggregator(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
        ],
        main_key=["wrong_key"],
    )
    msg = (
        "Got main_key=['wrong_key'], but column not in "
        "['userId', 'movieId', 'rating', 'genre']."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        join_agg.check_cols(main)

    # check too many main keys
    join_agg = JoinAggregator(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
        ],
        main_key=["userId", "movieId"],
    )
    msg = "The number of main keys must be either 1 or match the number of tables"
    with pytest.raises(ValueError, match=msg):
        join_agg.check_cols(main)

    # check main key length
    join_agg = JoinAggregator(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
            (main, ["userId"], ["rating", "genre"]),
        ],
        main_key="userId",
    )
    join_agg.check_cols(main)
    assert join_agg.main_keys_ == ["userId", "userId"]

    # check missing agg or keys cols in tables
    join_agg = JoinAggregator(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
            (main, ["movieId"], ["wrong_key"]),
        ],
        main_key=["userId", "movieId"],
    )
    msg = "{'wrong_key'} are missing in table 2"
    with pytest.raises(ValueError, match=msg):
        join_agg.check_cols(main)

    # check no suffix with one table
    join_agg = JoinAggregator(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
        ],
        main_key=["userId"],
    )
    join_agg.check_cols(main)
    assert join_agg.suffixes_ == [""]

    # check no suffix with multiple tables
    join_agg = JoinAggregator(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
            (main, ["movieId"], ["rating"]),
        ],
        main_key=["userId"],
    )
    join_agg.check_cols(main)
    assert join_agg.suffixes_ == ["_1", "_2"]

    # check incorrect suffix type
    join_agg = JoinAggregator(
        tables=[
            (main, ["userId"], ["rating", "genre"]),
            (main, ["movieId"], ["rating"]),
        ],
        main_key=["userId"],
        suffixes=1,
    )
    msg = "Suffixes must be a list of string matching the number of tables, got: '1'"
    with pytest.raises(ValueError, match=msg):
        join_agg.check_cols(main)

    # check inconsistent number of suffixes
    join_agg = JoinAggregator(
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
        join_agg.check_cols(main)


def test_join_agg_default_ops():
    # check default agg_ops
    join_agg = JoinAggregator(
        tables=[
            (main, "userId", ["rating", "genre"]),
        ],
        main_key=["userId"],
    )
    join_agg.fit(main)
    assert join_agg.agg_ops_ == ["mean", "mode"]

    # check invariant agg_ops input
    join_agg = JoinAggregator(
        tables=[
            (main, "userId", ["rating", "genre"]),
        ],
        main_key=["userId"],
        agg_ops=["min", "max", "mode"],
    )
    join_agg.fit(main)
    assert join_agg.agg_ops_ == ["min", "max", "mode"]

    # check not supported agg_ops
    join_agg = JoinAggregator(
        tables=[
            (main, "userId", ["rating", "genre"]),
        ],
        main_key=["userId"],
        agg_ops=["most_frequent", "mode"],
    )
    msg = (
        "'agg_ops' options are ['sum', 'mean', 'std', 'min', 'max', 'mode']"
        ", got: 'most_frequent'."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        join_agg.fit(main)

    # check split ops
    num_ops, categ_ops = split_num_categ_ops(
        ["mean", "std", "min", "mode", "max", "sum"]
    )
    assert num_ops == ["mean", "std", "min", "max", "sum"]
    assert categ_ops == ["mode"]


def test_dispatch_assembling():
    engine = dispatch_assembling_engine(
        [
            (main, None, None),
            (main, None, None),
        ]
    )
    assert isinstance(engine, PandasAssemblingEngine)

    msg = "Only Pandas or Polars dataframes are currently supported"
    with pytest.raises(NotImplementedError, match=msg):
        dispatch_assembling_engine(
            [
                (main, None, None),
                (main.values, None, None),
            ]
        )

    # XXX add Polars check here when supported
