import numpy as np
import pandas as pd

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


def test_histogram_narrow_range():
    # Non-regression for https://github.com/skrub-data/skrub/issues/2189
    # float32 columns with adjacent values smaller than bin precision
    low = np.float32(10.0)
    high = np.nextafter(low, 11.0)
    data = pd.Series([low, high])
    _, hist = _plotting.histogram(data)
    assert hist["bin_counts"].sum() == 2
    assert len(hist["bin_edges"]) == 2
