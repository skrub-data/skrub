from datetime import timedelta

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import skrub._dataframe as sbd
from skrub._duration_to_float import DurationToFloat, duration_to_float
from skrub.core import RejectColumn


def test_duration_to_float(df_module):
    df = df_module.make_column(
        "duration",
        [
            timedelta(seconds=3600),
            timedelta(milliseconds=123),
            timedelta(days=1),
            timedelta(microseconds=456),
            None,
        ],
    )

    expected = df_module.make_column(
        "duration", [3600.0, 0.123, 86400.0, 0.000456, None]
    )

    transformer = DurationToFloat()
    transformed = transformer.fit_transform(df)
    assert_array_almost_equal(sbd.to_numpy(transformed), sbd.to_numpy(expected))


def test_duration_to_float_rejects_non_duration(df_module):
    df = df_module.make_column("not_duration", [1, 2, 3, 4])
    transformer = DurationToFloat()
    with pytest.raises(RejectColumn, match="Expected a duration column, got*"):
        transformer.fit_transform(df)


def test_dispatched_duration_to_float(df_module):
    s = df_module.make_column(
        "", [timedelta(days=1), timedelta(hours=1), timedelta(microseconds=1)]
    )
    out = duration_to_float(s)
    assert_array_almost_equal(
        sbd.to_numpy(out),
        np.array([86400.0, 3600.0, 1e-6]),
    )
