import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pandas.testing
import pytest


def _example_data_dict():
    return {
        "int-col": [4, 0, -1, None],
        "int-not-null-col": [4, 0, -1, 10],
        "float-col": [4.5, 0.5, None, -1.5],
        "str-col": ["one", None, "three", "four"],
        "bool-col": [False, True, None, True],
        "bool-not-null-col": [False, True, True, True],
        "datetime-col": [
            datetime.datetime.fromisoformat(dt)
            for dt in [
                "2020-02-03T12:30:05",
                "2021-03-15T00:37:15",
                "2022-02-13T17:03:25",
            ]
        ] + [None],
        "date-col": [
            datetime.date.fromisoformat(dt)
            for dt in ["2002-02-03", "2001-05-17", "2005-02-13", "2004-10-02"]
        ],
    }


_DATAFAME_MODULES_INFO = {}
_DATAFAME_MODULES_INFO["pandas-numpy-dtypes"] = SimpleNamespace(
    **{
        "name": "pandas",
        "description": "pandas-numpy-dtypes",
        "module": pd,
        "DataFrame": pd.DataFrame,
        "Column": pd.Series,
        "make_dataframe": pd.DataFrame.from_dict,
        "make_column": lambda name, values: pd.Series(name=name, data=values),
        "assert_frame_equal": pandas.testing.assert_frame_equal,
        "assert_column_equal": pandas.testing.assert_series_equal,
        "empty_dataframe": pd.DataFrame(),
        "empty_column": pd.Series([], dtype="object"),
        "example_dataframe": pd.DataFrame(_example_data_dict()),
        "example_column": pd.Series(
            _example_data_dict()["float-col"], name="float-col"
        ),
        "dtypes": {
            "float32": np.float32,
            "float64": np.float64,
            "int32": np.int32,
            "int64": np.int64,
        },
    }
)

_DATAFAME_MODULES_INFO["pandas-nullable-dtypes"] = SimpleNamespace(
    **{
        "name": "pandas",
        "description": "pandas-nullable-dtypes",
        "module": pd,
        "DataFrame": pd.DataFrame,
        "Column": pd.Series,
        "make_dataframe": lambda data: pd.DataFrame(data).convert_dtypes(),
        "make_column": lambda name, values: pd.Series(
            name=name, data=values
        ).convert_dtypes(),
        "assert_frame_equal": pandas.testing.assert_frame_equal,
        "assert_column_equal": pandas.testing.assert_series_equal,
        "empty_dataframe": pd.DataFrame(),
        "empty_column": pd.Series([], dtype="object"),
        "example_dataframe": pd.DataFrame(_example_data_dict()).convert_dtypes(),
        "example_column": pd.Series(
            _example_data_dict()["float-col"], name="float-col"
        ).convert_dtypes(),
        "dtypes": {
            "float32": pd.Float32Dtype(),
            "float64": pd.Float64Dtype(),
            "int32": pd.Int32Dtype(),
            "int64": pd.Int64Dtype(),
        },
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
            "description": "polars",
            "module": pl,
            "DataFrame": pl.DataFrame,
            "Column": pl.Series,
            "make_dataframe": pl.from_dict,
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
            "dtypes": {
                "float32": pl.Float32,
                "float64": pl.Float64,
                "int32": pl.Int32,
                "int64": pl.Int64,
            },
        }
    )


pd.set_option("display.show_dimensions", False)


@pytest.fixture
def all_dataframe_modules():
    return _DATAFAME_MODULES_INFO


@pytest.fixture(params=list(_DATAFAME_MODULES_INFO.keys()))
def df_module(request):
    """Return information about a dataframe module (either polars or pandas).

    Information is accessed through attributes, for example ``df_module.name``,
    ``df_module.make_dataframe(dict(a=[1, 2, 3, 4]))``.

    The fixture is parametrized with the installed dataframe modules so a test
    that requests it will be executed once for each installed module (eg once
    for pandas and once for polars if both are installed).

    The list of provided attributes is:

    name
        The module name as a string.
    module
        The module object itself (either ``pandas`` or ``polars``).
    DataFrame
        The module's dataframe class.
    Column
        The module's column class (``pd.Series`` or ``pl.Series``).
    make_dataframe
        A function that takes a dictionary of ``{column_name: column_values}``
        and returns a dataframe.
    make_column
        A function that takes a name and sequence of values and returns a column.
        ``df_module.make_column("country", ["France", "Spain"])``
    assert_frame_equal
        A function that asserts 2 dataframes are equal.
    assert_column_equal
        A function that asserts 2 columns are equal.
    empty_dataframe
        A dataframe with 0 rows and 0 columns.
    empty_lazyframe
        A lazy dataframe with 0 rows and 0 columns (only for polars).
    empty_column
        A column of length 0.
    example_dataframe
        An example dataframe, see ``_example_data_dict`` in this module for the
        contents.
    example_column
        An example column; the "float-col" column from the example dataframe.
    dtypes
        A mapping from dtype names to types, keys are:
        ['float32', 'float64', 'int32', 'int64'].
    """
    return _DATAFAME_MODULES_INFO[request.param]


@pytest.fixture
def example_data_dict():
    return _example_data_dict()


@pytest.fixture(params=[False, True])
def use_fit_transform(request):
    """A simple fixture that yields False, then True.

    The benefit is to reduce the number of times we have to parametrize a test
    manually.
    """
    return request.param
