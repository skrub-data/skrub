from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from skrub._datetime_encoder import TIME_LEVELS, DatetimeEncoder


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


def get_constant_date(as_array=False):
    df = pd.DataFrame(
        [
            ["2020-01-01", "2020-02-04", "2021-02-05"],
            ["2020-01-01", "2020-02-04", "2021-02-05"],
            ["2020-01-01", "2020-02-04", "2021-02-05"],
            ["2020-01-01", "2020-02-04", "2021-02-05"],
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
            ["2022-01-01 23:23:43", "2020-12-25 11:12:00", pd.NaT],
            ["2023-02-03 11:12:12", "2020-02-04 08:32:00", "2023-02-05 23:00:00"],
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


@pytest.mark.parametrize("as_array", [True, False])
@pytest.mark.parametrize(
    "get_data_func, features",
    [
        (get_date, TIME_LEVELS[: TIME_LEVELS.index("day") + 1]),
        (get_datetime, TIME_LEVELS),
        (get_tz_datetime, TIME_LEVELS),
        (get_nanoseconds, TIME_LEVELS),
    ],
)
@pytest.mark.parametrize(
    "add_total_second, add_day_of_the_week",
    list(product([True, False], [True, False])),
)
@pytest.mark.parametrize("extract_until", TIME_LEVELS)
def test_fit(
    as_array,
    get_data_func,
    features,
    add_total_second,
    add_day_of_the_week,
    extract_until,
):
    X = get_data_func(as_array=as_array)
    enc = DatetimeEncoder(
        add_day_of_the_week=add_day_of_the_week,
        add_total_second=add_total_second,
        extract_until=extract_until,
    )
    enc.fit(X)

    total_second = ["total_second"] if add_total_second else []
    day_of_week = ["day_of_week"] if add_day_of_the_week else []

    if extract_until in features:
        features_ = features[: features.index(extract_until) + 1]
    else:
        features_ = deepcopy(features)

    features_ += total_second + day_of_week
    columns = range(X.shape[1])
    expected_features_per_column = {col: features_ for col in columns}

    expected_format_per_column = {col: np.asarray(X)[0, col] for col in columns}

    expected_n_features_out = sum(
        len(val) for val in expected_features_per_column.values()
    )

    expected_feature_names = [
        f"{col}_{feature}" for col in columns for feature in features_
    ]

    assert enc.features_per_column_ == expected_features_per_column
    assert enc.format_per_column_ == expected_format_per_column
    assert enc.n_features_out_ == expected_n_features_out
    assert enc.get_feature_names_out() == expected_feature_names


def test_format_nan():
    X = get_nan_datetime()
    enc = DatetimeEncoder().fit(X)
    expected_format_per_column = {
        0: "2020-01-01 10:12:01",
        1: "2020-02-04 22:12:00",
        2: "2020-01-03 10:00:00",
    }
    assert enc.format_per_column_ == expected_format_per_column


def test_format_nz():
    X = get_tz_datetime()
    enc = DatetimeEncoder().fit(X)
    assert enc.format_per_column_ == {0: "2020-01-01 10:12:01+05:30"}


def test_extract_until_none():
    X = get_datetime()
    enc = DatetimeEncoder(
        extract_until=None,
        add_total_second=False,
    )
    enc.fit(X)

    assert enc.features_per_column_ == {0: [], 1: [], 2: []}
    assert enc.n_features_out_ == 0
    assert enc.get_feature_names_out() == []


def test_transform_date():
    X = get_date()
    enc = DatetimeEncoder(
        add_total_second=False,
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


def test_transform_datetime():
    X = get_datetime()
    enc = DatetimeEncoder(
        extract_until="second",
        add_total_second=False,
    )
    X_trans = enc.fit_transform(X)
    X_trans_expected = np.array(
        [
            [2020, 1, 1, 10, 12, 1, 2020, 1, 2, 10, 23, 0, 2020, 1, 3, 10, 0, 0],
            [2021, 2, 3, 12, 45, 23, 2020, 2, 4, 22, 12, 0, 2021, 2, 5, 12, 0, 0],
            [2022, 1, 1, 23, 23, 43, 2020, 12, 25, 11, 12, 0, 2022, 1, 3, 11, 0, 0],
            [2023, 2, 3, 11, 12, 12, 2020, 2, 4, 8, 32, 0, 2023, 2, 5, 23, 0, 0],
        ]
    )
    assert_array_equal(X_trans, X_trans_expected)


def test_transform_tz():
    X = get_tz_datetime()
    enc = DatetimeEncoder(
        add_total_second=True,
    )
    X_trans = enc.fit_transform(X)
    X_trans_expected = np.array(
        [
            [2020, 1, 1, 10, 1.57785372e09],
            [2021, 2, 3, 12, 1.61233652e09],
            [2022, 1, 1, 23, 1.64105962e09],
            [2023, 2, 3, 11, 1.67540293e09],
        ]
    )
    assert_allclose(X_trans, X_trans_expected)


def test_transform_nan():
    X = get_nan_datetime()
    enc = DatetimeEncoder(
        add_total_second=True,
    )
    X_trans = enc.fit_transform(X)
    X_trans_expected = np.array(
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
            [
                2023,
                2,
                3,
                11,
                1.67542273e09,
                2020,
                2,
                4,
                8,
                1.58080512e09,
                2023,
                2,
                5,
                23,
                1.67563800e09,
            ],
        ]
    )
    assert_allclose(X_trans, X_trans_expected)
