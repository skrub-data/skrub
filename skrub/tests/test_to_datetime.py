from datetime import timezone

import pandas as pd
import pytest
from sklearn.utils.fixes import parse_version

from skrub import _dataframe as sbd
from skrub._dispatch import dispatch
from skrub._on_each_column import OnEachColumn, RejectColumn
from skrub._to_datetime import ToDatetime, to_datetime

ISO = "%Y-%m-%dT%H:%M:%S"


@dispatch
def strftime(col, format):
    raise NotImplementedError()


@strftime.specialize("pandas")
def _(col, format):
    return col.dt.strftime(format)


@strftime.specialize("polars")
def _(col, format):
    return col.dt.strftime(format)


def to_iso(dt_col):
    return sbd.to_list(strftime(dt_col, ISO))


@pytest.fixture
def datetime_col(df_module):
    return sbd.col(df_module.example_dataframe, "datetime-col")


@pytest.mark.parametrize(
    "format",
    [
        "%Y%m%dT%H%M%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y%m%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%G-W%V-%u",
    ],
)
@pytest.mark.parametrize("provide_format", [False, True])
def test_string_to_datetime(df_module, datetime_col, format, provide_format):
    if not provide_format and (
        format == "%G-W%V-%u"
        or (
            parse_version(pd.__version__) < parse_version("2.0.0")
            and format in ["%Y%m%dT%H%M%S", "%Y-%m-%dT%H:%M:%S.%f"]
        )
    ):
        pytest.xfail(
            "TODO improve datetime parsing, pandas does not find some ISO formats."
        )
    as_str = strftime(datetime_col, format)
    if provide_format:
        encoder = ToDatetime(format=format)
    else:
        encoder = ToDatetime()
    as_dt = encoder.fit_transform(as_str)
    assert encoder.format_ == format
    df_module.assert_column_equal(strftime(as_dt, format), as_str)


def test_datetime_to_datetime(datetime_col):
    encoder = ToDatetime(format="xyz")
    assert encoder.fit_transform(datetime_col) is datetime_col
    assert encoder.format_ is None


def test_rejected_columns(df_module):
    with pytest.raises(ValueError, match=".*does not contain strings"):
        ToDatetime().fit_transform(sbd.col(df_module.example_dataframe, "float-col"))
    with pytest.raises(ValueError, match="Could not find a datetime format"):
        ToDatetime().fit_transform(sbd.col(df_module.example_dataframe, "str-col"))
    with pytest.raises(ValueError, match="Failed .* using the format '%d/%m/%Y'"):
        ToDatetime(format="%d/%m/%Y").fit_transform(
            strftime(sbd.col(df_module.example_dataframe, "datetime-col"), "%Y-%m-%d")
        )


def test_transform_failures(datetime_col, df_module):
    encoder = ToDatetime().fit(strftime(datetime_col, ISO))
    test_col = df_module.make_column(
        "", ["2020-01-02T00:20:23", "01/02/2020", "xyz", None]
    )
    transformed = encoder.transform(test_col)
    expected = ["2020-01-02T00:20:23", "????", "????", "????"]
    assert sbd.to_list(sbd.fill_nulls(strftime(transformed, ISO), "????")) == expected


def test_mixed_offsets(df_module):
    s = df_module.make_column(
        "when", ["2020-01-02T08:00:01+02:00", "2020-01-02T08:00:01+04:00"]
    )
    encoder = ToDatetime()
    transformed = encoder.fit_transform(s)
    assert encoder.output_time_zone_ == "UTC"
    assert to_iso(transformed) == [
        "2020-01-02T06:00:01",
        "2020-01-02T04:00:01",
    ]


def localize(df_module, dt_col, time_zone):
    if df_module.name == "pandas":
        return dt_col.dt.tz_localize(time_zone)
    assert df_module.name == "polars"
    return dt_col.dt.replace_time_zone(time_zone)


def convert(df_module, dt_col, time_zone):
    if df_module.name == "pandas":
        return dt_col.dt.tz_convert(time_zone)
    assert df_module.name == "polars"
    return dt_col.dt.convert_time_zone(time_zone)


def test_fit_aware_transform_naive(df_module, datetime_col):
    aware = localize(df_module, datetime_col, "Europe/Paris")
    encoder = ToDatetime()
    encoder.fit(aware)
    assert encoder.output_time_zone_ == "Europe/Paris"
    assert to_iso(encoder.transform(datetime_col)) == to_iso(
        convert(df_module, localize(df_module, datetime_col, "UTC"), "Europe/Paris")
    )


def test_fit_naive_transform_aware(df_module, datetime_col):
    aware = localize(df_module, datetime_col, "Europe/Paris")
    encoder = ToDatetime()
    encoder.fit(datetime_col)
    assert encoder.output_time_zone_ is None
    assert to_iso(encoder.transform(aware)) == to_iso(convert(df_module, aware, "UTC"))


@pytest.mark.parametrize(
    "source", ["Europe/Paris", "America/Sao_Paulo", "UTC", timezone.utc]
)
@pytest.mark.parametrize(
    "dest", ["Europe/Paris", "America/Sao_Paulo", "UTC", timezone.utc]
)
def test_transform_from_a_different_timezone(df_module, datetime_col, source, dest):
    if timezone.utc in (source, dest) and df_module.name == "polars":
        # polars only receives strings as time zones.
        # for pandas we also test with timezone.utc because it results in
        # series where .dt.tz is a standard library timezone rather than a
        # pandas timezone, so it does not have the 'zone' attribute and we must
        # use its '.tzname' method instead.
        return
    fit_col = localize(df_module, datetime_col, source)
    encoder = ToDatetime().fit(fit_col)
    transform_col = convert(df_module, fit_col, dest)
    assert to_iso(encoder.transform(transform_col)) == to_iso(fit_col)


def test_fit_object_column():
    col = pd.Series(["2020-02-01T00:01:02", True])
    with pytest.raises(RejectColumn, match="Could not find a datetime format"):
        ToDatetime().fit(col)


@pytest.mark.parametrize("time_zone", [None, "UTC", "America/Toronto", "Asia/Istanbul"])
def test_polars_date_columns(all_dataframe_modules, time_zone):
    pl = pytest.importorskip("polars")
    datetime = (
        pl.Series(["2020-02-01T12:01:02"])
        .str.to_datetime()
        .dt.replace_time_zone(time_zone)
    )
    date = datetime.cast(pl.Date)
    encoder = ToDatetime().fit(date)
    assert_equal = all_dataframe_modules["polars"].assert_column_equal
    assert_equal(encoder.transform(datetime), date)
    encoder = ToDatetime().fit(datetime)
    out = encoder.transform(date)
    if time_zone is None:
        assert_equal(out, datetime.dt.truncate("1d"))
    else:
        assert_equal(
            out,
            datetime.dt.truncate("1d")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone(time_zone),
        )


def test_to_datetime_func(df_module, datetime_col):
    with pytest.raises(TypeError, match=".*must be .* Series or DataFrame"):
        to_datetime("2020-02-01T00:01:02")
    df_module.assert_column_equal(
        to_datetime(datetime_col), ToDatetime().fit_transform(datetime_col)
    )
    cols = (
        ("datetime-col",)
        if df_module.name == "pandas"
        else ("datetime-col", "date-col")
    )
    df_module.assert_frame_equal(
        to_datetime(df_module.example_dataframe),
        OnEachColumn(ToDatetime(), cols=cols).fit_transform(
            df_module.example_dataframe
        ),
    )
    float_col = df_module.example_column
    assert sbd.is_float(float_col)
    assert to_datetime(float_col) is float_col
