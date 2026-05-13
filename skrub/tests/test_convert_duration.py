from datetime import timedelta

import pytest
from numpy.testing import assert_array_almost_equal

from skrub._convert_duration import ConvertDuration
from skrub.core import RejectColumn


def test_convert_duration(df_module):
    df = df_module.make_column(
        "duration",
        [
            timedelta(seconds=3600),
            timedelta(milliseconds=123),
            timedelta(days=1),
            timedelta(microseconds=456),
        ],
    )

    expected = df_module.make_column("duration", [3600.0, 0.123, 86400.0, 0.000456])

    transformer = ConvertDuration()
    transformed = transformer.fit_transform(df)
    assert_array_almost_equal(transformed, expected)


def test_convert_duration_rejects_non_duration(df_module):
    df = df_module.make_column("not_duration", [1, 2, 3, 4])
    transformer = ConvertDuration()
    with pytest.raises(RejectColumn, match="Expected a duration column, got*"):
        transformer.fit_transform(df)
