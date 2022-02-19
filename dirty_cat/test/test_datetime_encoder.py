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

def get_dirty_datetime_array():
    return np.array([pd.to_datetime(["2020-01-01 10:12:01", "2020-01-02 10:23:00", "2020-01-03 10:00:00"]),
                  pd.to_datetime([np.nan, "2020-02-04 22:12:00", "2021-02-05 12:00:00"]),
                  pd.to_datetime(["2022-01-01 23:23:43", "2020-12-25 11:12:00", pd.NaT]),
                  pd.to_datetime(["2023-02-03 11:12:12", "2020-02-04 08:32:00", "2023-02-05 23:00:00"])])

def get_datetime_with_TZ_array():
    res = pd.DataFrame([pd.to_datetime(["2020-01-01 10:12:01"]),
                  pd.to_datetime(["2021-02-03 12:45:23"]),
                  pd.to_datetime(["2022-01-01 23:23:43"]),
                  pd.to_datetime(["2023-02-03 11:12:12"])])
    for col in res.columns:
        res[col] = pd.DatetimeIndex(res[col]).tz_localize("Asia/Kolkata")
    return res

def test_fit():
    # Dates
    X = get_date_array()
    enc = DatetimeEncoder()
    expected_to_extract_full = ["year", "month", "day", "hour", "other"]
    expected_features_per_column = {0: ["year", "month", "day"],
                           1: ["month", "day"],
                           2: ["year", "month", "day"]}
    enc.fit(X)
    assert enc.to_extract_full == expected_to_extract_full
    assert enc.features_per_column == expected_features_per_column

    X = get_date_array()
    enc = DatetimeEncoder(add_day_of_the_week=True, add_holidays=True)
    expected_to_extract_full = ["year", "month", "day", "hour", "other"]
    expected_features_per_column = {0: ["year", "month", "day", "dayofweek", "holiday"],
                           1: ["month", "day", "dayofweek", "holiday"],
                           2: ["year", "month", "day", "dayofweek", "holiday"]}
    enc.fit(X)
    assert enc.to_extract_full == expected_to_extract_full
    assert enc.features_per_column == expected_features_per_column


    # Datetimes
    X = get_datetime_array()
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_to_extract_full = ["year", "month", "day", "hour", "other"]
    expected_features_per_column = {0: ["year", "month", "day", "hour", "other", "dayofweek"],
                           1: ["month", "day", "hour", "other", "dayofweek"],
                           2: ["year", "month", "day", "hour", "dayofweek"]}
    enc.fit(X)
    assert enc.to_extract_full == expected_to_extract_full
    assert enc.features_per_column == expected_features_per_column

    X = get_datetime_array()
    enc = DatetimeEncoder(extract_until="minute")
    expected_to_extract_full = ["year", "month", "day", "hour", "minute", "other"]
    expected_features_per_column = {0: ["year", "month", "day", "hour", "minute", "other"],
                           1: ["month", "day", "hour", "minute"],
                           2: ["year", "month", "day", "hour"]}
    enc.fit(X)
    assert enc.to_extract_full == expected_to_extract_full
    assert enc.features_per_column == expected_features_per_column

    # Dirty Datetimes
    X = get_dirty_datetime_array()
    enc = DatetimeEncoder()
    expected_to_extract_full = ["year", "month", "day", "hour", "other"]
    expected_features_per_column = {0: ["year", "month", "day", "hour", "other"],
                           1: ["month", "day", "hour", "other"],
                           2: ["year", "month", "day", "hour"]}
    enc.fit(X)
    assert enc.to_extract_full == expected_to_extract_full
    assert enc.features_per_column == expected_features_per_column

    # Datetimes with TZ
    X = get_datetime_with_TZ_array()
    enc = DatetimeEncoder()
    expected_to_extract_full = ["year", "month", "day", "hour", "other"]
    expected_features_per_column = {0: ["year", "month", "day", "hour", "other"]}
    enc.fit(X)
    assert enc.to_extract_full == expected_to_extract_full
    assert enc.features_per_column == expected_features_per_column

    # Feature names
    # Without column names
    X = get_datetime_array()
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_feature_names = ["0_year", "0_month", "0_day", "0_hour", "0_other", "0_dayofweek",
                              "1_month", "1_day", "1_hour", "1_other", "1_dayofweek",
                              "2_year", "2_month", "2_day", "2_hour", "2_dayofweek"]
    enc.fit(X)
    assert enc.get_feature_names() == expected_feature_names

    # With column names
    X = get_datetime_array()
    X = pd.DataFrame(X)
    X.columns = ["col1", "col2", "col3"]
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_feature_names = ["col1_year", "col1_month", "col1_day", "col1_hour", "col1_other", "col1_dayofweek",
                              "col2_month", "col2_day", "col2_hour", "col2_other", "col2_dayofweek",
                              "col3_year", "col3_month", "col3_day", "col3_hour", "col3_dayofweek"]
    enc.fit(X)
    assert enc.get_feature_names() == expected_feature_names



def test_transform():
    # Dates
    X = get_date_array()
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_result = np.array([[2020, 1, 1, 2, 1, 2, 3, 2020, 1, 3, 4],
                                [2021, 2, 3, 2, 2, 4, 1, 2021, 2, 5, 4],
                                [2022, 1, 1, 5, 12, 25, 4, 2022, 1, 3, 0],
                                [2023, 2, 3, 4, 2, 4, 1, 2023, 2, 5, 6]])
    enc.fit(X)
    assert np.allclose(enc.transform(X), expected_result, equal_nan=True)

    enc = DatetimeEncoder(add_day_of_the_week=False)
    expected_result = np.array([[2020, 1, 1, 1, 2, 2020, 1, 3],
                                [2021, 2, 3, 2, 4, 2021, 2, 5],
                                [2022, 1, 1, 12, 25, 2022, 1, 3],
                                [2023, 2, 3, 2, 4, 2023, 2, 5]])
    enc.fit(X)
    assert np.allclose(enc.transform(X), expected_result, equal_nan=True)

    enc = DatetimeEncoder(add_day_of_the_week=True, add_holidays=True)
    expected_result = np.array([[2020, 1, 1, 2, 1, 1, 2, 3, 0, 2020, 1, 3, 4, 0],
                                [2021, 2, 3, 2, 0, 2, 4, 1, 0, 2021, 2, 5, 4, 0],
                                [2022, 1, 1, 5, 0, 12, 25, 4, 1, 2022, 1, 3, 0, 0],
                                [2023, 2, 3, 4, 0, 2, 4, 1, 0, 2023, 2, 5, 6, 0]])
    enc.fit(X)
    assert np.allclose(enc.transform(X), expected_result, equal_nan=True)

    # Datetimes
    X = get_datetime_array()[:, 0].reshape(-1, 1)
    enc = DatetimeEncoder(add_day_of_the_week=True)
    # Check that the "other" feature is the right value and unit
    expected_result = np.array([[2020, 1, 1, 10, 12 / 60 + 1 / 3600, 2],
                                [2021, 2, 3, 12, 45 / 60 + 23 / 3600, 2],
                                [2022, 1, 1, 23, 23 / 60 + 43 / 3600, 5],
                                [2023, 2, 3, 11, 12 / 60 + 12 / 3600, 4]])
    enc.fit(X)
    X_trans = enc.transform(X)
    assert np.allclose(X_trans, expected_result, equal_nan=True)

    # Dirty Datetimes
    X = get_dirty_datetime_array()[:, 0].reshape(-1, 1)
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_result = np.array([[2020, 1, 1, 10, 12 / 60 + 1 / 3600, 2],
                                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                [2022, 1, 1, 23, 23 / 60 + 43 / 3600, 5],
                                [2023, 2, 3, 11, 12 / 60 + 12 / 3600, 4]])
    enc.fit(X)
    X_trans = enc.transform(X)
    assert np.allclose(X_trans, expected_result, equal_nan=True)

    # Datetimes with TZ
    # If the date are timezone-aware, all the feature extraction should be done
    # in the provided timezone.
    X = get_datetime_with_TZ_array()
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_result = np.array([[2020, 1, 1, 10, 12 / 60 + 1 / 3600, 2],
                                [2021, 2, 3, 12, 45 / 60 + 23 / 3600, 2],
                                [2022, 1, 1, 23, 23 / 60 + 43 / 3600, 5],
                                [2023, 2, 3, 11, 12 / 60 + 12 / 3600, 4]])
    enc.fit(X)
    X_trans = enc.transform(X)
    assert np.allclose(X_trans, expected_result, equal_nan=True)


