import datetime
from types import SimpleNamespace

import pandas as pd
import pandas.testing
import pytest


def _example_data_dict():
    return {
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
        "date-col": [
            datetime.date.fromisoformat(dt)
            for dt in ["2002-02-03", "2001-05-17", "2005-02-13", "2004-10-02"]
        ],
    }


_DATAFAME_MODULES_INFO = {}
_DATAFAME_MODULES_INFO["pandas"] = SimpleNamespace(
    **{
        "name": "pandas",
        "module": pd,
        "DataFrame": pd.DataFrame,
        "Column": pd.Series,
        "make_dataframe": pd.DataFrame,
        "make_column": lambda name, values: pd.Series(name=name, data=values),
        "assert_frame_equal": pandas.testing.assert_frame_equal,
        "assert_column_equal": pandas.testing.assert_series_equal,
        "empty_dataframe": pd.DataFrame(),
        "empty_column": pd.Series([], dtype="object"),
        "example_dataframe": pd.DataFrame(_example_data_dict()),
        "example_column": pd.Series(
            _example_data_dict()["float-col"], name="float-col"
        ),
    }
)

try:
    import polars as pl
    import polars.testing

    _POLARS_INSTALLED = True
except ImportError:
    _POLARS_INSTALLED = False

if _POLARS_INSTALLED:
    _DATAFAME_MODULES_INFO["polars"] = SimpleNamespace(
        **{
            "name": "polars",
            "module": pl,
            "DataFrame": pl.DataFrame,
            "Column": pl.Series,
            "make_dataframe": pl.DataFrame,
            "make_column": lambda name, values: pl.Series(name=name, values=values),
            "assert_frame_equal": polars.testing.assert_frame_equal,
            "assert_column_equal": polars.testing.assert_series_equal,
            "empty_dataframe": pl.DataFrame(),
            "empty_lazyframe": pl.DataFrame().lazy(),
            "empty_column": pl.Series(),
            "example_dataframe": pl.DataFrame(_example_data_dict()),
            "example_column": pl.Series(
                values=_example_data_dict()["float-col"], name="float-col"
            ),
        }
    )


pd.set_option("display.show_dimensions", False)


@pytest.fixture
def all_dataframe_modules():
    return _DATAFAME_MODULES_INFO


@pytest.fixture(params=list(_DATAFAME_MODULES_INFO.values()))
def df_module(request):
    return request.param


@pytest.fixture
def example_data_dict():
    return _example_data_dict()
