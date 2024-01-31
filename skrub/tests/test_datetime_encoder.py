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
from skrub._datetime_encoder import _TIME_LEVELS, DatetimeEncoder
from skrub._to_datetime import to_datetime

MODULES = [pd]
ASSERT_TUPLES = [(pd, assert_frame_equal)]


if POLARS_SETUP:
    import polars as pl
    from polars.testing import assert_frame_equal as assert_frame_equal_pl

    MODULES.append(pl)
    ASSERT_TUPLES.append((pl, assert_frame_equal_pl))


def get_date(as_array=False):
    df = pd.DataFrame(
        [
            ["2020-01-01", "2020-01-02", "2020-01-03"],
            ["2021-02-03", "2020-02-04", "2021-02-05"],
            ["2022-01-01", "2020-12-25", "2022-01-03"],
            ["2023-02-03", "2020-02-04", "2023-02-05"],
        ],
    )
    df.columns = list(map(str, df.columns))
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
    df.columns = list(map(str, df.columns))
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
    df.columns = list(map(str, df.columns))
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
    df.columns = list(map(str, df.columns))
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
    df.columns = list(map(str, df.columns))
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


@pytest.mark.parametrize("px", MODULES)
@pytest.mark.parametrize("as_array", [True, False])
@pytest.mark.parametrize(
    "get_data_func, features, format",
    [
        (get_date, _TIME_LEVELS[: _TIME_LEVELS.index("day") + 1], "%Y-%m-%d"),
        (get_datetime, _TIME_LEVELS, "%Y-%m-%d %H:%M:%S"),
        (get_tz_datetime, _TIME_LEVELS, "%Y-%m-%d %H:%M:%S%z"),
        (get_nanoseconds, _TIME_LEVELS, "%Y-%m-%d %H:%M:%S.%f"),
    ],
)
@pytest.mark.parametrize(
    "add_total_seconds, add_day_of_the_week",
    list(product([True, False], [True, False])),
)
@pytest.mark.parametrize("resolution", _TIME_LEVELS)
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
    day_of_week = ["day_of_the_week"] if add_day_of_the_week else []

    if resolution in features:
        features_ = features[: features.index(resolution) + 1]
    else:
        features_ = deepcopy(features)

    features_ += total_seconds + day_of_week
    columns = range(X.shape[1])

    try:
        columns = X.columns
    except AttributeError:
        columns = list(map(str, range(X.shape[1])))
    expected_input_to_outputs = {
        col: [f"{col}_{f}" for f in features_] for col in columns
    }
    expected_n_features_out = len(features_) * X.shape[1]
    expected_feature_names = [
        f"{col}_{feature}" for col in columns for feature in features_
    ]

    assert enc.input_to_outputs_ == expected_input_to_outputs
    assert enc.n_features_out_ == expected_n_features_out
    assert_array_equal(enc.get_feature_names_out(), expected_feature_names)


@pytest.mark.parametrize("px", MODULES)
@pytest.mark.parametrize(
    "get_data_func, expected_datetime_columns",
    [
        (get_date, ["0", "1", "2"]),
        (get_datetime, ["0", "1", "2"]),
        (get_tz_datetime, ["0"]),
    ],
)
@pytest.mark.parametrize("random_state", np.arange(20))
def test_to_datetime(px, get_data_func, expected_datetime_columns, random_state):
    if is_module_polars(px):
        pytest.xfail(reason="AssertionError is raised when using Polars.")
    X = get_data_func()
    X = to_datetime(X)
    X = px.DataFrame(X)
    datetime_columns = [col for col in X.columns if is_datetime64_any_dtype(X[col])]
    assert_array_equal(datetime_columns, expected_datetime_columns)


@pytest.mark.parametrize("px", MODULES)
def test_format_nan(px):
    X = get_nan_datetime()
    X = px.DataFrame(X)
    enc = DatetimeEncoder().fit(X)
    expected_datetime_formats = {
        "0": "%Y-%m-%d %H:%M:%S",
        "1": "%Y-%m-%d %H:%M:%S",
        "2": "%Y-%m-%d %H:%M:%S",
    }
    assert enc.datetime_formats_ == expected_datetime_formats


@pytest.mark.parametrize("px", MODULES)
def test_format_nz(px):
    X = get_tz_datetime()
    X = px.DataFrame(X)
    enc = DatetimeEncoder().fit(X)
    assert enc.datetime_formats_ == {"0": "%Y-%m-%d %H:%M:%S%z"}


@pytest.mark.parametrize("px", MODULES)
def test_resolution_none(px):
    X = get_datetime()
    px.DataFrame(X)
    enc = DatetimeEncoder(
        resolution=None,
        add_total_seconds=False,
    )
    enc.fit(X)

    assert enc.input_to_outputs_ == {"0": [], "1": [], "2": []}
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
    assert enc.datetime_formats_ == {"a": "%Y-%m-%d", "e": "%d/%m/%Y"}

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


@pytest.mark.parametrize("px", MODULES)
def test_datetime_encoder_invalid_params(px):
    X = get_datetime()
    X = px.DataFrame(X)

    with pytest.raises(ValueError, match=r"(?=.*'resolution' options)"):
        DatetimeEncoder(resolution="hello").fit(X)

    DatetimeEncoder(resolution=None).fit(X)


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
    with pytest.raises(TypeError, match=".*must be a pandas or polars Series.*"):
        assert_array_equal(to_datetime(X), X)


def test_to_datetime_type_error():
    # 3d tensor
    X = [[["2021-01-01"]]]
    with pytest.raises(TypeError):
        to_datetime(X)


def test_to_datetime_invalid_params():
    with pytest.raises(TypeError, match=r".*unexpected keyword argument"):
        to_datetime(2020, errors="skip")

    with pytest.raises(TypeError, match=r".*unexpected keyword argument"):
        to_datetime(2020, unit="second")


def test_only_ambiguous():
    X_col = pd.Series(["2021/10/10", "2020/01/02"])
    out = to_datetime(X_col)
    # monthfirst by default
    expected_out = np.array(["2021-10-10", "2020-01-02"], dtype="datetime64[ns]")
    assert_array_equal(out, expected_out)


def test_monthfirst_only():
    X_col = pd.Series(["2021/02/02", "2021/01/15"])
    out = to_datetime(X_col)
    expected_out = np.array(["2021-02-02", "2021-01-15"], dtype="datetime64[ns]")
    assert_array_equal(out, expected_out)


def test_preserve_dtypes():
    X = get_mixed_type_dataframe()
    X["b"] = X["b"].astype("category")
    non_datetime_columns = ["b", "c", "f"]

    X_trans = to_datetime(X)
    assert_frame_equal(X_trans[non_datetime_columns], X[non_datetime_columns])
