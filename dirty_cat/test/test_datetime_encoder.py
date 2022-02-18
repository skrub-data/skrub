import pandas as pd
from dirty_cat.datetime_encoder import DatetimeEncoder
import numpy as np


def get_date_array():
    return np.array([pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                  pd.to_datetime(["2021-02-03", "2020-02-04", "2021-02-05"]),
                  pd.to_datetime(["2022-01-01", "2020-12-25", "2022-01-03"]),
                  pd.to_datetime(["2023-02-03", "2020-02-04", "2023-02-05"])])

def get_datetime_array():
    return np.array([pd.to_datetime(["2020-01-01 10:12:01", "2020-01-02 10:23:00", "2020-01-03 10:00:00"]),
                  pd.to_datetime(["2021-02-03 12:45:23", "2020-02-04 22:12:00", "2021-02-05 12:00:00"]),
                  pd.to_datetime(["2022-01-01 23:23:43", "2020-12-25 11:12:00", "2022-01-03 11:00:00"]),
                  pd.to_datetime(["2023-02-03 11:12:12", "2020-02-04 08:32:00", "2023-02-05 23:00:00"])])

def test_fit():
    # Dates
    X = get_date_array()
    enc = DatetimeEncoder()
    expected_to_extract_full = ["year", "month", "day", "hour", "other"]
    expected_to_extract = {0: ["year", "month", "day", "dayofweek"],
                           1: ["month", "day", "dayofweek"],
                           2: ["year", "month", "day", "dayofweek"]}
    enc.fit(X)
    assert enc.to_extract_full == expected_to_extract_full
    assert enc.to_extract == expected_to_extract

    X = get_date_array()
    enc = DatetimeEncoder(add_holidays=True)
    expected_to_extract_full = ["year", "month", "day", "hour", "other"]
    expected_to_extract = {0: ["year", "month", "day", "dayofweek", "holiday"],
                           1: ["month", "day", "dayofweek", "holiday"],
                           2: ["year", "month", "day", "dayofweek", "holiday"]}
    enc.fit(X)
    assert enc.to_extract_full == expected_to_extract_full
    assert enc.to_extract == expected_to_extract


    # Datetimes
    X = get_datetime_array()
    enc = DatetimeEncoder()
    expected_to_extract_full = ["year", "month", "day", "hour", "other"]
    expected_to_extract = {0: ["year", "month", "day", "hour", "other", "dayofweek"],
                           1: ["month", "day", "hour", "other", "dayofweek"],
                           2: ["year", "month", "day", "hour", "dayofweek"]}
    enc.fit(X)
    assert enc.to_extract_full == expected_to_extract_full
    assert enc.to_extract == expected_to_extract

    X = get_datetime_array()
    enc = DatetimeEncoder(extract_until="minute")
    expected_to_extract_full = ["year", "month", "day", "hour", "minute", "other"]
    expected_to_extract = {0: ["year", "month", "day", "hour", "minute", "other", "dayofweek"],
                           1: ["month", "day", "hour", "minute", "dayofweek"],
                           2: ["year", "month", "day", "hour", "dayofweek"]}
    enc.fit(X)
    assert enc.to_extract_full == expected_to_extract_full
    assert enc.to_extract == expected_to_extract


def test_transform():
    # Dates
    X = get_date_array()
    enc = DatetimeEncoder()
    expected_result = np.array([[2020, 1, 1, 2, 1, 2, 3, 2020, 1, 3, 4],
                                [2021, 2, 3, 2, 2, 4, 1, 2021, 2, 5, 4],
                                [2022, 1, 1, 5, 12, 25, 4, 2022, 1, 3, 0],
                                [2023, 2, 3, 4, 2, 4, 1, 2023, 2, 5, 6]])
    enc.fit(X)
    assert np.array_equal(enc.transform(X), expected_result)

    enc = DatetimeEncoder(add_day_of_the_week=False)
    expected_result = np.array([[2020, 1, 1, 1, 2, 2020, 1, 3],
                                [2021, 2, 3, 2, 4, 2021, 2, 5],
                                [2022, 1, 1, 12, 25, 2022, 1, 3],
                                [2023, 2, 3, 2, 4, 2023, 2, 5]])
    enc.fit(X)
    assert np.array_equal(enc.transform(X), expected_result)

    enc = DatetimeEncoder(add_holidays=True)
    expected_result = np.array([[2020, 1, 1, 2, 1, 1, 2, 3, 0, 2020, 1, 3, 4, 0],
                                [2021, 2, 3, 2, 0, 2, 4, 1, 0, 2021, 2, 5, 4, 0],
                                [2022, 1, 1, 5, 0, 12, 25, 4, 1, 2022, 1, 3, 0, 0],
                                [2023, 2, 3, 4, 0, 2, 4, 1, 0, 2023, 2, 5, 6, 0]])
    enc.fit(X)
    assert np.array_equal(enc.transform(X), expected_result)

    #Datetimes
    X = get_datetime_array()[:, 0].reshape(-1, 1)
    enc = DatetimeEncoder()
    # Check that the "other" feature is the right value and unit
    expected_result = np.array([[2020, 1, 1, 10, 12 / 60 + 1 / 3600, 2],
                                [2021, 2, 3, 12, 45 / 60 + 23 / 3600, 2],
                                [2022, 1, 1, 23, 23 / 60 + 43 / 3600, 5],
                                [2023, 2, 3, 11, 12 / 60 + 12 / 3600, 4]])
    enc.fit(X)
    X_trans = enc.transform(X)
    assert np.allclose(X_trans, expected_result)
