"""
Tests generating.py (synthetic dataset generation).
"""

import numpy as np
import pandas as pd
import pytest

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

    df_nulls = toy_cities(nulls=1, size=10)
    df_no_nulls = toy_cities(nulls=0, size=10)
    assert pd.isnull(df_nulls["cities"]).all()
    assert not pd.isnull(df_no_nulls["cities"]).any()


_SEED_MESSAGE = "seed must be a positive integer"
_SIZE_MESSAGE = "size must be a positive integer"
_NULLS_MESSAGE = "nulls must be a number"
_METRICS_MESSAGE = "n_metrics must be a positive integer"


@pytest.mark.parametrize(
    ("seed", "size", "n_metrics", "nulls", "message"),
    [
        ("yes", 10, 4, 0.1, _SEED_MESSAGE),
        (-5, 10, 4, 0.1, _SEED_MESSAGE),
        (25, None, 4, 0.1, _SIZE_MESSAGE),
        (25, -8, 4, 0.1, _SIZE_MESSAGE),
        (25, 10, 4.5, 0.1, _METRICS_MESSAGE),
        (25, 10, -5, 0.1, _METRICS_MESSAGE),
        (25, 10, 4, "nulls", _NULLS_MESSAGE),
        (25, 10, 4, 3, _NULLS_MESSAGE),
    ],
)
def test_toy_cities_errors(seed, size, n_metrics, nulls, message):
    with pytest.raises(ValueError, match=message):
        _ = toy_cities(seed=seed, size=size, nulls=nulls, n_metrics=n_metrics)
