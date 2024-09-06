import json

import numpy as np
import pytest

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
        (102.0, "102."),
        (10_045, "10,045"),
        (np.int32(10_045), "10,045"),
        (np.float32(10_045), "1.00e+04"),
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


def test_json_encoder():
    x = np.ones(1, dtype="int32")
    y = np.ones(1, dtype="float32")
    d = {"a": x[0], "b": y[0]}
    assert json.dumps(d, cls=_utils.JSONEncoder) == '{"a": 1, "b": 1.0}'
    with pytest.raises(TypeError, match=".*JSON serializable"):
        json.dumps({"a": x}, cls=_utils.JSONEncoder)
