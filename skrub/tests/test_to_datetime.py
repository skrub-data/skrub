import pytest

from skrub import _dataframe as sbd
from skrub._dispatch import dispatch
from skrub._to_datetime import ToDatetime

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
        "%Y%m%d%H%M%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y%m%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%G-W%V-%u",
    ],
)
@pytest.mark.parametrize("provide_format", [False, True])
def test_string_to_datetime(df_module, datetime_col, format, provide_format):
    if format == "%G-W%V-%u" and not provide_format:
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
        "", ["2020-01-02T00:20:23", "2020-01-02 00:20:23", "xyz", None]
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
