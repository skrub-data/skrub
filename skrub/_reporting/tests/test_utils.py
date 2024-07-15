import json

import numpy as np
import pytest

from skrub import _dataframe as sbd
from skrub._reporting import _utils


@pytest.mark.parametrize(
    "s_in, s_out",
    [
        (1, 1),
        ("aa", "aa"),
        ("a" * 120, "a" * 70 + "[…50 more chars]"),
    ],
)
def test_ellide_string(s_in, s_out):
    assert _utils.ellide_string(s_in) == s_out


def test_ellide_string_short():
    assert _utils.ellide_string_short("a" * 100) == "a" * 29 + "…"


@pytest.mark.parametrize(
    "n_in, n_out",
    [
        (102, "102"),
        ("abc", "abc"),
        (102.22222222222, "102."),
        (1.0222222222222, "1.02"),
        (0.10222222222222, "0.102"),
    ],
)
def test_format_number(n_in, n_out):
    assert _utils.format_number(n_in) == n_out


@pytest.mark.parametrize(
    "n_in, n_out",
    [
        (1.0, "100.0%"),
        (0.22222222, "22.2%"),
        (0.002, "0.2%"),
        (0.0001, "<\u202f0.1%"),
    ],
)
def test_format_percent(n_in, n_out):
    assert _utils.format_percent(n_in) == n_out


@pytest.mark.parametrize(
    "value, expected",
    [
        (10, [3]),
        (None, [4, 5]),
        ([10, 20], [1, 2, 3]),
    ],
)
def test_filter_snippet(df_module, value, expected):
    df = df_module.make_dataframe(
        {"the column": [20, 20, 10, None, None], "x": [1, 2, 3, 4, 5]}
    )
    if df_module.name == "polars":
        pl = df_module.module
    else:
        pd = df_module.module
    if isinstance(value, list):
        snippet = _utils.filter_isin_snippet(value, "the column", df_module.name)
    else:
        snippet = _utils.filter_equal_snippet(value, "the column", df_module.name)
    res = eval(snippet, locals())
    assert sbd.to_list(sbd.col(res, "x")) == expected


def test_filter_snippet_unkown_lib():
    assert (
        _utils.filter_isin_snippet([0], "the column", "xxx")
        == "Unknown dataframe library: xxx"
    )
    assert (
        _utils.filter_equal_snippet(0, "the column", "xxx")
        == "Unknown dataframe library: xxx"
    )


def test_json_encoder():
    x = np.ones(1, dtype="int32")
    y = np.ones(1, dtype="float32")
    d = {"a": x[0], "b": y[0]}
    assert json.dumps(d, cls=_utils.JSONEncoder) == '{"a": 1, "b": 1.0}'
    with pytest.raises(TypeError, match=".*JSON serializable"):
        json.dumps({"a": x}, cls=_utils.JSONEncoder)
