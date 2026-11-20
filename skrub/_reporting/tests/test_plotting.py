from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from skrub._reporting import _plotting


def test_histogram():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    o = rng.uniform(-100, 100, size=10)

    data = pd.Series(np.concatenate([x, o]))
    _, hist = _plotting.histogram(data)
    assert (hist["n_low_outliers"], hist["n_high_outliers"]) == (5, 4)

    data = pd.Series(np.concatenate([x, o - 1000]))
    _, hist = _plotting.histogram(data)
    assert (hist["n_low_outliers"], hist["n_high_outliers"]) == (10, 0)

    data = pd.Series(np.concatenate([x, o + 1000]))
    _, hist = _plotting.histogram(data)
    assert (hist["n_low_outliers"], hist["n_high_outliers"]) == (0, 10)

    data = pd.Series(x)
    _, hist = _plotting.histogram(data)
    assert (hist["n_low_outliers"], hist["n_high_outliers"]) == (0, 0)

    data = pd.Series([0.0])
    _, hist = _plotting.histogram(data)
    assert (hist["n_low_outliers"], hist["n_high_outliers"]) == (0, 0)

    # Test a column with values very close to each other.
    # There are less than 10 (default number of histogram bins) representable
    # floating-point numbers in the range, which would cause numpy or
    # matplotlib histogram to raise an exception without appropriate handling
    # in skrub.
    low = np.float32(10.0)
    high = np.nextafter(low, 11.0)

    data = pd.Series([low, high])
    _, hist = _plotting.histogram(data)
    assert hist["n_low_outliers"] == 0
    assert hist["n_high_outliers"] == 0

    # all infinite. +- inf are outliers
    data = pd.Series(
        [None, None, float("nan"), float("-inf"), float("-inf"), float("inf")]
    )
    _, hist = _plotting.histogram(data)
    assert hist["n_low_outliers"] == 2
    assert hist["n_high_outliers"] == 1


def test_histogram_decimal_column():
    # Decimal columns (polars Decimal, pandas pyarrow decimal) used to raise
    # TypeError in _get_range because np.quantile cannot multiply
    # decimal.Decimal values by floats.
    values = [Decimal("1.50"), None, Decimal("2.00"), Decimal("3.25")]
    _, expected = _plotting.histogram(pd.Series([1.5, 2.0, 3.25]))

    pl = pytest.importorskip("polars")
    col_pl = pl.Series(values, dtype=pl.Decimal(10, 2))
    _, hist_pl = _plotting.histogram(col_pl)
    assert hist_pl["bin_counts"].tolist() == expected["bin_counts"].tolist()
    assert (
        hist_pl["n_low_outliers"],
        hist_pl["n_high_outliers"],
    ) == (
        expected["n_low_outliers"],
        expected["n_high_outliers"],
    )

    pa = pytest.importorskip("pyarrow")
    col_pd = pd.Series(values, dtype=pd.ArrowDtype(pa.decimal128(10, 2)))
    _, hist_pd = _plotting.histogram(col_pd)
    assert hist_pd["bin_counts"].tolist() == expected["bin_counts"].tolist()
    assert (
        hist_pd["n_low_outliers"],
        hist_pd["n_high_outliers"],
    ) == (
        expected["n_low_outliers"],
        expected["n_high_outliers"],
    )
