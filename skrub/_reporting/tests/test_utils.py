import datetime
import json

import numpy as np
import pytest

from skrub import _dataframe as sbd
from skrub._reporting import _utils


@pytest.mark.parametrize(
    "s_in, s_out",
    [
        (1, "1"),
        ("aa", "aa"),
        ("a\na", "a a"),
        ("a" * 70, "a" * 30 + "…\u200e"),
        ("0" * 70, "0" * 30 + "…"),
        # Note: in the right-to-left examples below, the ellipsis of the
        # expected output probably appears to be on the wrong side on your
        # display. This is because the Unicode marks (eg U+061C ARABIC LETTER MARK)
        # are written as Unicode escape sequences "u061c" rather than literal
        # characters in the python strings. If we include them literally the
        # "…" will be on the right side but some editors or github might issue
        # a warning about the potential malicious use of bidirectional
        # characters:
        # https://en.wikipedia.org/wiki/Trojan_Source
        #
        # To visually check the output of TableReport for some right-to-left
        # samples, see:
        # ``/skrub/_reporting/js_tests/make-reports``.
        (
            (
                "اللغة هي نسق على من الإشارات والرموز، تشكل أداة من أدوات المعرفة،"
                " وتعتبر اللغة أهم وسائل التفاهم والاحتكاك بين أفراد المجتمع في جميع"
                " ميادين الحياة"
            ),
            "اللغة هي نسق على من الإشارات و…\u061c",
        ),
        (
            (
                "שפה היא דרך תקשורת המבוססת על מערכת"
                " סמלים מורכבת בעלת חוקיות, המאפשרת לקודד"
            ),
            "שפה היא דרך תקשורת המבוססת על…\u200f",
        ),
    ],
)
def test_ellide_string(s_in, s_out):
    assert _utils.ellide_string(s_in) == s_out


def test_ellide_string_empty():
    # useless corner case to make codecov happy
    assert _utils.ellide_string(" a", 1) == "…"


def test_ellide_non_string():
    # non-regression for #1195: objects in columns must be converted to strings
    # before elliding and plotting
    class A:
        def __repr__(self):
            return "one\ntwo\nthree"

    assert _utils.ellide_string(A()) == "one two three"


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


def test_svg_to_img_src():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'fill="currentColor" viewBox="0 0 16 16">'
        '<path fill-rule="evenodd" d="M1 8a.5.5 0 0 1 '
        ".5-.5h11.793l-3.147-3.146a.5.5 0 0 1 "
        ".708-.708l4 4a.5.5 0 0 1 0 .708l-4 4a.5.5 "
        '0 0 1-.708-.708L13.293 8.5H1.5A.5.5 0 0 1 1 8"/></svg>'
    )
    assert (
        _utils.svg_to_img_src(svg)
        == "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL"
        "3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9ImN1cnJlbnRDb2xvciI"
        "gdmlld0JveD0iMCAwIDE2IDE2Ij48cGF0aCBmaWxsLXJ1bGU9ImV2Z"
        "W5vZGQiIGQ9Ik0xIDhhLjUuNSAwIDAgMSAuNS0uNWgxMS43OTNsLTM"
        "uMTQ3LTMuMTQ2YS41LjUgMCAwIDEgLjcwOC0uNzA4bDQgNGEuNS41I"
        "DAgMCAxIDAgLjcwOGwtNCA0YS41LjUgMCAwIDEtLjcwOC0uNzA4TDE"
        "zLjI5MyA4LjVIMS41QS41LjUgMCAwIDEgMSA4Ii8+PC9zdmc+"
    )


@pytest.mark.parametrize(
    "kwargs,value,unit",
    [
        ({"seconds": 2e-6}, 2, "microsecond"),
        ({"seconds": 0.5}, 500, "millisecond"),
        ({"seconds": 5}, 5, "second"),
        ({"hours": 5}, 5, "hour"),
        ({"days": 5}, 5, "day"),
        ({"days": 500}, 1.3689, "year"),
    ],
)
def test_duration_to_numeric(df_module, kwargs, value, unit):
    s = df_module.make_column("", [datetime.timedelta(**kwargs)])
    chosen_value, chosen_unit = _utils.duration_to_numeric(s)
    assert sbd.to_list(chosen_value)[0] == pytest.approx(value, 0.001)
    assert chosen_unit == unit
