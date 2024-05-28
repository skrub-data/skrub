from types import SimpleNamespace

import pytest

from skrub import DatetimeEncoder
from skrub import _dataframe as sbd
from skrub import _selectors as s
from skrub._on_each_column import OnEachColumn
from skrub._to_float32 import ToFloat32


def date(df_module):
    return sbd.to_datetime(
        df_module.make_column("when", ["2020-01-01", None, "2022-01-01"]),
        "%Y-%m-%d",
    )


def datetime(df_module):
    return sbd.to_datetime(
        df_module.make_column(
            "when",
            ["2020-01-01 10:12:01", None, "2022-01-01 23:23:43"],
        ),
        "%Y-%m-%d %H:%M:%S",
    )


def tz_datetime(df_module):
    # The equivalent dtype is "datetime64[ns, Asia/Kolkata]"
    col = sbd.to_datetime(
        df_module.make_column(
            "when",
            [
                "2020-01-01 10:12:01",
                None,
                "2022-01-01 23:23:43",
            ],
        ),
        "%Y-%m-%d %H:%M:%S",
    )
    if df_module.name == "pandas":
        return col.dt.tz_localize("Asia/Kolkata")
    else:
        assert df_module.name == "polars"
        return col.dt.replace_time_zone("Asia/Kolkata")


def nanoseconds(df_module):
    return sbd.to_datetime(
        df_module.make_column(
            "when",
            [
                "2020-01-01 10:12:01.123456789",
                None,
                "2022-01-01 23:23:43.123987654",
            ],
        ),
        (
            "%Y-%m-%d %H:%M:%S%.f"
            if df_module.name == "polars"
            else "%Y-%m-%d %H:%M:%S.%f"
        ),
    )


_DATETIME_COLUMN_GENERATORS = {
    k: globals()[k] for k in ["date", "datetime", "tz_datetime", "nanoseconds"]
}


@pytest.fixture
def datetime_cols(df_module):
    return SimpleNamespace(
        **{k: v(df_module) for k, v in _DATETIME_COLUMN_GENERATORS.items()}
    )


@pytest.fixture(params=list(_DATETIME_COLUMN_GENERATORS.keys()))
def a_datetime_col(request, df_module):
    return _DATETIME_COLUMN_GENERATORS[request.param](df_module)


@pytest.fixture
def expected_features(df_module):
    values = {
        "when_year": [2020.0, None, 2022.0],
        "when_month": [1.0, None, 1.0],
        "when_day": [1.0, None, 1.0],
        "when_hour": [10.0, None, 23.0],
        "when_minute": [12.0, None, 23.0],
        "when_second": [1.0, None, 43.0],
        "when_microsecond": [123456.0, None, 123987.0],
        "when_nanosecond": [789.0, None, 654.0],
        "when_total_seconds": [1577873536.0, None, 1641079424.0],
        "when_weekday": [3.0, None, 6.0],
    }
    res = df_module.make_dataframe(values)
    return OnEachColumn(ToFloat32()).fit_transform(res)


def test_fit_transform(a_datetime_col, expected_features, df_module, use_fit_transform):
    enc = DatetimeEncoder()
    if use_fit_transform:
        res = enc.fit_transform(a_datetime_col)
    else:
        res = enc.fit(a_datetime_col).transform(a_datetime_col)
    expected_features = s.select(
        expected_features,
        [f"when_{f}" for f in enc.extracted_features_],
    )
    df_module.assert_frame_equal(res, expected_features, rtol=1e-4)


@pytest.mark.parametrize(
    "params, extracted_features",
    [
        (dict(), ["year", "month", "day", "hour", "total_seconds"]),
        (dict(resolution="month"), ["year", "month", "total_seconds"]),
        (
            dict(resolution="nanosecond"),
            [
                "year",
                "month",
                "day",
                "hour",
                "minute",
                "second",
                "microsecond",
                "nanosecond",
                "total_seconds",
            ],
        ),
        (dict(resolution=None), ["total_seconds"]),
        (dict(add_total_seconds=False), ["year", "month", "day", "hour"]),
        (
            dict(add_weekday=True, add_total_seconds=False),
            ["year", "month", "day", "hour", "weekday"],
        ),
    ],
)
def test_extracted_features_choice(datetime_cols, params, extracted_features):
    enc = DatetimeEncoder(**params)
    res = enc.fit_transform(datetime_cols.datetime)
    assert enc.extracted_features_ == extracted_features
    assert sbd.column_names(res) == [f"when_{f}" for f in enc.extracted_features_]


def test_time_not_extracted_from_date_col(datetime_cols):
    enc = DatetimeEncoder(resolution="nanosecond")
    enc.fit(datetime_cols.date)
    assert enc.extracted_features_ == ["year", "month", "day", "total_seconds"]


def test_invalid_resolution(datetime_cols):
    with pytest.raises(ValueError, match=r".*'resolution' options are"):
        DatetimeEncoder(resolution="hello").fit(datetime_cols.datetime)


def test_reject_non_datetime(df_module):
    with pytest.raises(ValueError, match=".*does not have Date or Datetime dtype."):
        DatetimeEncoder().fit_transform(df_module.example_column)
