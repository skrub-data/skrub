import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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


def test_plots_do_not_show_in_interactive_mode():
    """Plotting functions must not display figures even when matplotlib is in
    interactive mode (e.g. inside a Jupyter notebook with %matplotlib inline).
    All figures are serialized to SVG and embedded in the HTML report."""
    rng = np.random.default_rng(0)
    data = pd.Series(rng.normal(size=50))
    plt.ion()
    try:
        before = plt.get_fignums()
        _plotting.histogram(data)
        after = plt.get_fignums()
        assert before == after, "histogram() left open figure(s) in interactive mode"
    finally:
        plt.ioff()
