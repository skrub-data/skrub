import numpy as np
import pandas as pd

from skrub._reporting import _plotting


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
