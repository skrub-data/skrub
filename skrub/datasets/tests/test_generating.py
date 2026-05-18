"""
Tests generating.py (synthetic dataset generation).
"""

import numpy as np
import pandas as pd

from skrub.datasets._generating import make_deduplication_data, toy_cities


def test_make_deduplication_data():
    np.random.seed(123)
    assert make_deduplication_data(["abc", "cba", "test1"], [3, 2, 1], 0.3) == [
        "agr",
        "abc",
        "abc",
        "cba",
        "cba",
        "test1",
    ]
    assert make_deduplication_data(["abc", "cba", "test1"], [1, 2, 3], 0.8) == [
        "pbc",
        "pza",
        "cba",
        "erxt1",
        "test1",
        "test1",
    ]


def test_toy_cities():
    df = toy_cities(seed=25, size=10)
    assert len(df.columns) == 9
    assert len(df["uid"]) == 10
    expected = pd.DataFrame(
        {
            "cities": [
                "Budapest",
                None,
                "Helsinki",
                "Warsaw",
                "Amsterdam",
                "Madrid",
                "Prague",
                None,
                "Vienna",
                "Sofia",
            ]
        }
    )
    pd.testing.assert_series_equal(df["cities"], expected["cities"])
