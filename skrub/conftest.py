import datetime

import pandas as pd

_DATAFRAME_TYPES = [pd.DataFrame]
_COLUMN_TYPES = [pd.Series]

try:
    import polars as pl

    _POLARS_INSTALLED = True
    _DATAFRAME_TYPES.append(pl.DataFrame)
    # may add LazyFrame in the future
    _COLUMN_TYPES.append(pl.Series)
except ImportError:
    _POLARS_INSTALLED = False
    pass

import pytest

from skrub._dataframe import make_dataframe_like

pd.set_option("display.show_dimensions", False)


@pytest.fixture
def empty_pandas_dataframe():
    return pd.DataFrame()


@pytest.fixture
def empty_polars_dataframe():
    if not _POLARS_INSTALLED:
        pytest.skip("Polars not installed in current environment.")
    return pl.DataFrame()


@pytest.fixture
def empty_pandas_series():
    return pd.Series(dtype="object")


@pytest.fixture
def empty_polars_series():
    if not _POLARS_INSTALLED:
        pytest.skip("Polars not installed in current environment.")
    return pl.Series()


@pytest.fixture(params=_DATAFRAME_TYPES)
def empty_df(request):
    return request.param()


@pytest.fixture(params=_COLUMN_TYPES)
def empty_column(request):
    if request.param is pd.Series:
        return pd.Series(dtype="object")
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
            for dt in [
                "2020-02-03T12:30:05",
                "2021-03-15T00:37:15",
                "2022-02-13T17:03:25",
            ]
        ]
        + [None],
        "datetime": [
            datetime.date.fromisoformat(dt)
            for dt in ["2002-02-03", "2001-05-17", "2005-02-13"]
        ]
        + [None],
    }
    return make_dataframe_like(empty_df, data)
