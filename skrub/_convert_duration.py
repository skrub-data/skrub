"""This transformer converts durations to seconds."""

from . import _dataframe as sbd
from ._single_column_transformer import RejectColumn, SingleColumnTransformer


class ConvertDuration(SingleColumnTransformer):
    """Convert duration columns to seconds.

    This transformer converts duration columns to seconds. It only works on
    columns with duration dtype. Other dtypes are rejected.

    For ``pandas``, only columns with ``duration`` dtype are modified:

    Examples
    --------

    >>> import pandas as pd
    >>> from datetime import timedelta
    >>> from skrub._convert_duration import ConvertDuration
    >>> s = pd.Series([
    ...     timedelta(seconds=3600), timedelta(minutes=2), timedelta(days=1)
    ... ])
    >>> s
    0   0 days 01:00:00
    1   0 days 00:02:00
    2   1 days 00:00:00
    dtype: timedelta64[...]
    >>> converter = ConvertDuration()
    >>> converter.fit_transform(s)
    0    3600.0
    1    120.0
    2    86400.0
    dtype: float64

    Columns that do not have ``duration`` dtype are rejected:

    >>> s = pd.Series(['1 day', '2 days', '3 days'], name='s')
    >>> converter.fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub.core.RejectColumn: Expected a duration column, got...

    For ``polars``, only columns with ``Duration`` dtype are modified:

    >>> import pytest
    >>> pl = pytest.importorskip('polars')
    >>> s = pl.Series([
    ...     timedelta(seconds=3600), timedelta(minutes=2), timedelta(days=1)
    ... ])
    >>> s
    shape: (3,)
    Series: '' [duration[μs]]
    [
        1h
        2m
        1d
    ]
    >>> converter.fit_transform(s)
    shape: (3,)
    Series: '' [f64]
    [
        3600.0
        120.0
        86400.0
    ]
    >>> s = pl.Series('s', ['1 day', '2 days', '3 days'], dtype=pl.String)
    >>> converter.fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub.core.RejectColumn: Expected a duration column, got String
    """

    def fit_transform(self, col, y=None):
        del y
        if not sbd.is_duration(col):
            raise RejectColumn(f"Expected a duration column, got {col.dtype}")
        return self.transform(col)

    def transform(self, col, y=None):
        del y

        column = sbd.convert_duration(col)
        return column
