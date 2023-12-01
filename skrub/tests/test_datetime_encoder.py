from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.api.types import is_datetime64_any_dtype
from pandas.testing import assert_frame_equal

from skrub._dataframe._polars import POLARS_SETUP
from skrub._dataframe._test_utils import is_module_polars
from skrub._datetime_encoder import (
    TIME_LEVELS,
    DatetimeEncoder,
    _is_pandas_format_mixed_available,
    to_datetime,
)

MODULES = [pd]
ASSERT_TUPLES = [(pd, assert_frame_equal)]

if POLARS_SETUP:
    import polars as pl
    from polars.testing import assert_frame_equal as assert_frame_equal_pl

    MODULES.append(pl)
    ASSERT_TUPLES.append((pl, assert_frame_equal_pl))

NANOSECONDS_FORMAT = (
    "%Y-%m-%d %H:%M:%S.%f" if _is_pandas_format_mixed_available() else None
)
MSG_MIN_PANDAS_SKIP = "Pandas format=mixed is not available"


def get_date(as_array=False):
    df = pd.DataFrame(
        [
            ["2020-01-01", "2020-01-02", "2020-01-03"],
            ["2021-02-03", "2020-02-04", "2021-02-05"],
            ["2022-01-01", "2020-12-25", "2022-01-03"],
            ["2023-02-03", "2020-02-04", "2023-02-05"],
        ],
    )
    if as_array:
        return df.to_numpy()
    return df


def get_datetime(as_array=False):
    df = pd.DataFrame(
        [
            ["2020-01-01 10:12:01", "2020-01-02 10:23:00", "2020-01-03 10:00:00"],
            ["2021-02-03 12:45:23", "2020-02-04 22:12:00", "2021-02-05 12:00:00"],
            ["2022-01-01 23:23:43", "2020-12-25 11:12:00", "2022-01-03 11:00:00"],
            ["2023-02-03 11:12:12", "2020-02-04 08:32:00", "2023-02-05 23:00:00"],
        ],
    )
    if as_array:
        return df.to_numpy()
    return df


def get_nanoseconds(as_array=False):
    df = pd.DataFrame(
        [
            ["2020-08-24 15:55:30.123456789", "2020-08-24 15:55:30.123456789"],
            ["2020-08-20 14:56:31.987654321", "2021-07-20 14:56:31.987654321"],
            ["2020-08-20 14:57:32.123987654", "2023-09-20 14:57:32.123987654"],
            ["2020-08-20 14:58:33.987123456", "2023-09-20 14:58:33.987123456"],
        ],
    )
    if as_array:
        return df.to_numpy()
    return df


def get_nan_datetime(as_array=False):
    df = pd.DataFrame(
        [
            ["2020-01-01 10:12:01", None, "2020-01-03 10:00:00"],
            [np.nan, "2020-02-04 22:12:00", "2021-02-05 12:00:00"],
            ["2022-01-01 23:23:43", "2020-12-25 11:12:00", pd.NA],
        ],
    )
    if as_array:
        return df.to_numpy()
    return df


def get_tz_datetime(as_array=False):
    # The equivalent dtype is "datetime64[ns, Asia/Kolkata]"
    df = pd.DataFrame(
        [
            ["2020-01-01 10:12:01+05:30"],
            ["2021-02-03 12:45:23+05:30"],
            ["2022-01-01 23:23:43+05:30"],
            ["2023-02-03 11:12:12+05:30"],
        ],
    )
    if as_array:
        return df.to_numpy()
    return df


def get_mixed_type_dataframe():
    return pd.DataFrame(
        dict(
            a=["2020-01-01", "2020-02-04", "2021-02-05"],
            b=["yo", "ya", "yu"],
            c=[1, 2, 3],
            d=["1", "2", "3"],
            e=["01/01/2023", "03/01/2023", "14/01/2023"],
            f=[True, False, True],
        )
    )


def get_mixed_datetime_format(as_array=False):
    df = pd.DataFrame(
        dict(
            a=[
                "2022-10-15",
                "2021-12-25",
                "2020-05-18",
                "2019-10-15 12:00:00",
            ]
        )
    )
    if as_array:
        return df.to_numpy()
    return df


@pytest.mark.parametrize("px", MODULES)
@pytest.mark.parametrize("as_array", [True, False])
@pytest.mark.parametrize(
    "get_data_func, features, format",
    [
        (get_date, TIME_LEVELS[: TIME_LEVELS.index("day") + 1], "%Y-%m-%d"),
        (get_datetime, TIME_LEVELS, "%Y-%m-%d %H:%M:%S"),
        (get_tz_datetime, TIME_LEVELS, "%Y-%m-%d %H:%M:%S%z"),
        (get_nanoseconds, TIME_LEVELS, NANOSECONDS_FORMAT),
    ],
)
@pytest.mark.parametrize(
    "add_total_seconds, add_day_of_the_week",
    list(product([True, False], [True, False])),
)
@pytest.mark.parametrize("resolution", TIME_LEVELS)
def test_fit(
    px,
    as_array,
    get_data_func,
    features,
    format,
    add_total_seconds,
    add_day_of_the_week,
    resolution,
):
    X = get_data_func(as_array=as_array)
    enc = DatetimeEncoder(
        add_day_of_the_week=add_day_of_the_week,
        add_total_seconds=add_total_seconds,
        resolution=resolution,
    )
    enc.fit(X)

    total_seconds = ["total_seconds"] if add_total_seconds else []
    day_of_week = ["day_of_week"] if add_day_of_the_week else []

    if resolution in features:
        features_ = features[: features.index(resolution) + 1]
    else:
        features_ = deepcopy(features)

    features_ += total_seconds + day_of_week
    columns = range(X.shape[1])

    expected_index_to_features = {col: features_ for col in columns}
    expected_index_to_format = {col: format for col in columns}
    expected_n_features_out = len(features_) * X.shape[1]
    expected_feature_names = [
        f"{col}_{feature}" for col in columns for feature in features_
    ]

    assert enc.index_to_features_ == expected_index_to_features
    assert enc.index_to_format_ == expected_index_to_format
    assert enc.n_features_out_ == expected_n_features_out
    assert_array_equal(enc.get_feature_names_out(), expected_feature_names)


@pytest.mark.parametrize("px", MODULES)
@pytest.mark.parametrize(
    "get_data_func, expected_datetime_columns",
    [
        (get_date, [0, 1, 2]),
        (get_datetime, [0, 1, 2]),
        (get_tz_datetime, [0]),
        (get_mixed_type_dataframe, ["a", "e"]),
    ],
)
@pytest.mark.parametrize("random_state", np.arange(20))
def test_to_datetime(px, get_data_func, expected_datetime_columns, random_state):
    if is_module_polars(px):
        pytest.xfail(reason="AssertionError is raised when using Polars.")
    X = get_data_func()
    X = to_datetime(X, random_state=random_state)
    X = px.DataFrame(X)
    datetime_columns = [col for col in X.columns if is_datetime64_any_dtype(X[col])]
    assert_array_equal(datetime_columns, expected_datetime_columns)


@pytest.mark.parametrize("px", MODULES)
def test_format_nan(px):
    X = get_nan_datetime()
    X = px.DataFrame(X)
    enc = DatetimeEncoder().fit(X)
    expected_index_to_format = {
        0: "%Y-%m-%d %H:%M:%S",
        1: "%Y-%m-%d %H:%M:%S",
        2: "%Y-%m-%d %H:%M:%S",
    }
    assert enc.index_to_format_ == expected_index_to_format


@pytest.mark.parametrize("px", MODULES)
def test_format_nz(px):
    X = get_tz_datetime()
    X = px.DataFrame(X)
    enc = DatetimeEncoder().fit(X)
    assert enc.index_to_format_ == {0: "%Y-%m-%d %H:%M:%S%z"}


@pytest.mark.parametrize("px", MODULES)
def test_resolution_none(px):
    X = get_datetime()
    px.DataFrame(X)
    enc = DatetimeEncoder(
        resolution=None,
        add_total_seconds=False,
    )
    enc.fit(X)

    assert enc.index_to_features_ == {0: [], 1: [], 2: []}
    assert enc.n_features_out_ == 0
    assert_array_equal(enc.get_feature_names_out(), [])


@pytest.mark.parametrize("px", MODULES)
def test_transform_date(px):
    X = get_date()
    X = px.DataFrame(X)
    enc = DatetimeEncoder(
        add_total_seconds=False,
    )
    X_trans = enc.fit_transform(X)

    expected_result = np.array(
        [
            [2020, 1, 1, 2020, 1, 2, 2020, 1, 3],
            [2021, 2, 3, 2020, 2, 4, 2021, 2, 5],
            [2022, 1, 1, 2020, 12, 25, 2022, 1, 3],
            [2023, 2, 3, 2020, 2, 4, 2023, 2, 5],
        ]
    )
    X_trans = enc.transform(X)
    assert_array_equal(X_trans, expected_result)


@pytest.mark.parametrize("px", MODULES)
def test_transform_datetime(px):
    X = get_datetime()
    X = px.DataFrame(X)
    enc = DatetimeEncoder(
        resolution="second",
        add_total_seconds=False,
    )
    X_trans = enc.fit_transform(X)
    expected_X_trans = np.array(
        [
            [2020, 1, 1, 10, 12, 1, 2020, 1, 2, 10, 23, 0, 2020, 1, 3, 10, 0, 0],
            [2021, 2, 3, 12, 45, 23, 2020, 2, 4, 22, 12, 0, 2021, 2, 5, 12, 0, 0],
            [2022, 1, 1, 23, 23, 43, 2020, 12, 25, 11, 12, 0, 2022, 1, 3, 11, 0, 0],
            [2023, 2, 3, 11, 12, 12, 2020, 2, 4, 8, 32, 0, 2023, 2, 5, 23, 0, 0],
        ]
    )
    assert_array_equal(X_trans, expected_X_trans)


@pytest.mark.parametrize("px", MODULES)
def test_transform_tz(px):
    X = get_tz_datetime()
    X = px.DataFrame(X)
    enc = DatetimeEncoder(
        add_total_seconds=True,
    )
    X_trans = enc.fit_transform(X)
    expected_X_trans = np.array(
        [
            [2020, 1, 1, 10, 1.57785372e09],
            [2021, 2, 3, 12, 1.61233652e09],
            [2022, 1, 1, 23, 1.64105962e09],
            [2023, 2, 3, 11, 1.67540293e09],
        ]
    )
    assert_allclose(X_trans, expected_X_trans)


@pytest.mark.parametrize("px", MODULES)
def test_transform_nan(px):
    X = get_nan_datetime()
    X = px.DataFrame(X)
    enc = DatetimeEncoder(
        add_total_seconds=True,
    )
    X_trans = enc.fit_transform(X)
    expected_X_trans = np.array(
        [
            [
                2020,
                1,
                1,
                10,
                1.57787352e09,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                2020,
                1,
                3,
                10,
                1.57804560e09,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                2020,
                2,
                4,
                22,
                1.58085432e09,
                2021,
                2,
                5,
                12,
                1.61252640e09,
            ],
            [
                2022,
                1,
                1,
                23,
                1.64107942e09,
                2020,
                12,
                25,
                11,
                1.60889472e09,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        ]
    )
    assert_allclose(X_trans, expected_X_trans)


@pytest.mark.parametrize("px", MODULES)
def test_mixed_type_dataframe(px):
    if is_module_polars(px):
        pytest.xfail(
            reason=(
                "to_datetime(X) raises polars.exceptions.ComputeError: cannot cast"
                " 'Object' type"
            )
        )
    X = get_mixed_type_dataframe()
    X = px.DataFrame(X)
    enc = DatetimeEncoder().fit(X)
    assert enc.index_to_format_ == {0: "%Y-%m-%d", 4: "%d/%m/%Y"}

    X_dt = to_datetime(X)
    expected_dtypes = [
        np.dtype("<M8[ns]"),
        np.dtype("object"),
        np.dtype("int64"),
        np.dtype("object"),
        np.dtype("<M8[ns]"),
        np.dtype("bool"),
    ]
    assert X_dt.dtypes.to_list() == expected_dtypes

    X_dt = to_datetime(X.to_numpy())
    assert X_dt.dtype == np.object_


@pytest.mark.parametrize("px, assert_frame_equal_", ASSERT_TUPLES)
def test_indempotency(px, assert_frame_equal_):
    df = get_mixed_datetime_format()
    df = px.DataFrame(df)
    df_dt = to_datetime(df)
    df_dt_2 = to_datetime(df_dt)
    assert_frame_equal_(df_dt, df_dt_2)

    X_trans = DatetimeEncoder().fit_transform(df)
    X_trans_2 = DatetimeEncoder().fit_transform(df_dt)
    assert_array_equal(X_trans, X_trans_2)


@pytest.mark.parametrize("px", MODULES)
def test_datetime_encoder_invalid_params(px):
    X = get_datetime()
    X = px.DataFrame(X)

    with pytest.raises(ValueError, match=r"(?=.*'resolution' options)"):
        DatetimeEncoder(resolution="hello").fit(X)

    DatetimeEncoder(resolution=None).fit(X)

    with pytest.raises(ValueError, match=r"(?=.*'errors' options)"):
        DatetimeEncoder(errors="ignore").fit(X)


@pytest.mark.parametrize(
    "X",
    [
        True,
        "a",
        ["a", "b"],
        ("a", "b"),
        1,
        [1, 2],
        np.array([1, 2]),
        pd.Timestamp(2020, 1, 1),
        np.array([pd.Timestamp(2020, 1, 1), "hello"]),
        np.array(["2020-01-01", {"hello"}]),
        np.array(["2020-01-01", "hello", "2020-01-02"]),
    ],
)
def test_to_datetime_incorrect_skip(X):
    assert_array_equal(to_datetime(X), X)


def test_to_datetime_type_error():
    # 3d tensor
    X = [[["2021-01-01"]]]
    with pytest.raises(TypeError):
        to_datetime(X)


def test_to_datetime_invalid_params():
    with pytest.raises(ValueError, match=r"(?=.*errors options)"):
        to_datetime(2020, errors="skip")

    with pytest.raises(ValueError, match=r"(?=.*not a parameter of skrub)"):
        to_datetime(2020, unit="second")


@pytest.mark.skipif(
    not _is_pandas_format_mixed_available(),
    reason=MSG_MIN_PANDAS_SKIP,
)
def test_to_datetime_format_param():
    X_col = ["2021-01-01", "2021/01/01"]

    # without format (default)
    out = to_datetime(X_col)
    expected_out = np.array(["2021-01-01", "NaT"], dtype="datetime64[ns]")
    assert_array_equal(out, expected_out)

    # with format
    out = to_datetime(X_col, format="%Y/%m/%d")
    expected_out = np.array(["NaT", "2021-01-01"], dtype="datetime64[ns]")
    assert_array_equal(out, expected_out)


@pytest.mark.parametrize("px, assert_frame_equal_", ASSERT_TUPLES)
def test_mixed_datetime_format(px, assert_frame_equal_):
    df = get_mixed_datetime_format()
    df = px.DataFrame(df)

    df_dt = to_datetime(df)
    expected_df_dt = pd.DataFrame(
        dict(
            a=[
                pd.Timestamp("2022-10-15"),
                pd.Timestamp("2021-12-25"),
                pd.Timestamp("2020-05-18"),
                pd.Timestamp("2019-10-15 12:00:00"),
            ]
        )
    )
    expected_df_dt = px.DataFrame(expected_df_dt)
    assert_frame_equal_(df_dt, expected_df_dt)

    series_dt = to_datetime(df["a"])
    expected_series_dt = expected_df_dt["a"]
    assert_array_equal(series_dt, expected_series_dt)


@pytest.mark.skipif(not _is_pandas_format_mixed_available(), reason=MSG_MIN_PANDAS_SKIP)
def test_mix_of_unambiguous():
    X_col = ["2021/10/15", "01/14/2021"]
    out = to_datetime(X_col)
    expected_out = np.array(
        [np.datetime64("2021-10-15"), np.datetime64("NaT")],
        dtype="datetime64[ns]",
    )
    assert_array_equal(out, expected_out)


def test_only_ambiguous():
    X_col = ["2021/10/10", "2020/01/02"]
    out = to_datetime(X_col)
    # monthfirst by default
    expected_out = np.array(["2021-10-10", "2020-01-02"], dtype="datetime64[ns]")
    assert_array_equal(out, expected_out)


def test_monthfirst_only():
    X_col = ["2021/02/02", "2021/01/15"]
    out = to_datetime(X_col)
    expected_out = np.array(["2021-02-02", "2021-01-15"], dtype="datetime64[ns]")
    assert_array_equal(out, expected_out)


def test_preserve_dtypes():
    X = get_mixed_type_dataframe()
    X["b"] = X["b"].astype("category")
    non_datetime_columns = ["b", "c", "f"]

    X_trans = to_datetime(X)
    assert_frame_equal(X_trans[non_datetime_columns], X[non_datetime_columns])
