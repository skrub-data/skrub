import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from skrub._reporting import _plotting
from skrub._reporting._summarize import summarize_dataframe


def test_histogram():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    o = rng.uniform(-100, 100, size=10)

    data = pd.Series(np.concatenate([x, o]))
    _, n_low, n_high = _plotting.histogram(data)
    assert (n_low, n_high) == (5, 4)

    data = pd.Series(np.concatenate([x, o - 1000]))
    _, n_low, n_high = _plotting.histogram(data)
    assert (n_low, n_high) == (10, 0)

    data = pd.Series(np.concatenate([x, o + 1000]))
    _, n_low, n_high = _plotting.histogram(data)
    assert (n_low, n_high) == (0, 10)

    data = pd.Series(x)
    _, n_low, n_high = _plotting.histogram(data)
    assert (n_low, n_high) == (0, 0)

    data = pd.Series([0.0])
    _, n_low, n_high = _plotting.histogram(data)
    assert (n_low, n_high) == (0, 0)


@pytest.mark.skipif(
    "text.parse_math" not in plt.rcParams,
    reason="text.parse_math requires Matplotlib >= 3.6",
)
def test_value_counts_renders_dollar_wrapped_text_literally():
    svg = _plotting.value_counts([("$x + y$", 2), ("other", 1)], 2, 3)
    root = ET.fromstring(svg)
    text_labels = [
        "".join(element.itertext()).strip()
        for element in root.iter()
        if element.tag.endswith("text")
    ]

    assert "$x + y$" in text_labels


def test_summarize_dataframe_with_double_dollar_string_restores_rc_params():
    df = pd.DataFrame(
        {
            "text": [
                "this is not latex $$ just a double dollar sign",
                "another value",
            ]
        }
    )

    if "text.parse_math" not in plt.rcParams:
        summary = summarize_dataframe(
            df, with_plots=True, with_associations=False, verbose=0
        )
        assert summary["columns"][0]["value_counts_plot"].startswith("<?xml")
        return

    original_parse_math = plt.rcParams["text.parse_math"]
    plt.rcParams["text.parse_math"] = True
    try:
        summary = summarize_dataframe(
            df, with_plots=True, with_associations=False, verbose=0
        )
        assert summary["columns"][0]["value_counts_plot"].startswith("<?xml")
        assert plt.rcParams["text.parse_math"] is True
    finally:
        plt.rcParams["text.parse_math"] = original_parse_math
