import datetime

import pandas as pd

_DATAFRAME_TYPES = [pd.DataFrame]
try:
    import polars as pl

    _DATAFRAME_TYPES.append(pl.DataFrame)
    # may add LazyFrame in the future
except ImportError:
    pass

import pytest

from skrub._dataframe import dataframe_from_dict

pd.set_option("display.show_dimensions", False)


@pytest.fixture(params=_DATAFRAME_TYPES)
def empty_df(request):
    return request.param()


@pytest.fixture
def example_df(empty_df):
    data = {
        "int-col": [4, 0, -1, None],
        "float-col": [4.5, 0.5, None, -1.5],
        "str-col": ["one", None, "three", "four"],
        "bool-col": [False, True, None, True],
        "datetime-col": [
            datetime.datetime.fromisoformat(dt)
            for dt in ["2020-02-03T12:30:052021-15-03T11:37:152022-02-13T17:03:25"]
        ]
        + [None],
        "datetime": [
            datetime.date.fromisoformat(dt) for dt in ["2002-02-032001-15-032005-02-13"]
        ]
        + [None],
    }
    return dataframe_from_dict(empty_df, data)
