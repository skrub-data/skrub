import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from skrub._datetime_encoder import DatetimeEncoder


def get_date_array() -> np.array:
    return np.array(
        [
            pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            pd.to_datetime(["2021-02-03", "2020-02-04", "2021-02-05"]),
            pd.to_datetime(["2022-01-01", "2020-12-25", "2022-01-03"]),
            pd.to_datetime(["2023-02-03", "2020-02-04", "2023-02-05"]),
        ]
    )


def get_constant_date_array() -> np.array:
    return np.array(
        [
            pd.to_datetime(["2020-01-01", "2020-02-04", "2021-02-05"]),
            pd.to_datetime(["2020-01-01", "2020-02-04", "2021-02-05"]),
            pd.to_datetime(["2020-01-01", "2020-02-04", "2021-02-05"]),
            pd.to_datetime(["2020-01-01", "2020-02-04", "2021-02-05"]),
        ]
    )


def get_datetime_array() -> np.array:
    return np.array(
        [
            pd.to_datetime(
                ["2020-01-01 10:12:01", "2020-01-02 10:23:00", "2020-01-03 10:00:00"]
            ),
            pd.to_datetime(
                ["2021-02-03 12:45:23", "2020-02-04 22:12:00", "2021-02-05 12:00:00"]
            ),
            pd.to_datetime(
                ["2022-01-01 23:23:43", "2020-12-25 11:12:00", "2022-01-03 11:00:00"]
            ),
            pd.to_datetime(
                ["2023-02-03 11:12:12", "2020-02-04 08:32:00", "2023-02-05 23:00:00"]
            ),
        ]
    )


def get_dirty_datetime_array() -> np.array:
    return np.array(
        [
            np.array(
                pd.to_datetime(
                    [
                        "2020-01-01 10:12:01",
                        "2020-01-02 10:23:00",
                        "2020-01-03 10:00:00",
                    ]
                )
            ),
            np.array(
                pd.to_datetime([np.nan, "2020-02-04 22:12:00", "2021-02-05 12:00:00"])
            ),
            np.array(
                pd.to_datetime(["2022-01-01 23:23:43", "2020-12-25 11:12:00", pd.NaT])
            ),
            np.array(
                pd.to_datetime(
                    [
                        "2023-02-03 11:12:12",
                        "2020-02-04 08:32:00",
                        "2023-02-05 23:00:00",
                    ]
                )
            ),
        ]
    )


def get_datetime_with_TZ_array() -> pd.DataFrame:
    res = pd.DataFrame(
        [
            pd.to_datetime(["2020-01-01 10:12:01"]),
            pd.to_datetime(["2021-02-03 12:45:23"]),
            pd.to_datetime(["2022-01-01 23:23:43"]),
            pd.to_datetime(["2023-02-03 11:12:12"]),
        ]
    )
    for col in res.columns:
        res[col] = pd.DatetimeIndex(res[col]).tz_localize("Asia/Kolkata")
    return res


def test_fit() -> None:
    # Dates
    X = get_date_array()
    enc = DatetimeEncoder()
    expected_to_extract = ["year", "month", "day", "hour"]
    expected_features_per_column_ = {
        0: ["year", "month", "day"],
        1: ["month", "day"],
        2: ["year", "month", "day"],
    }
    enc.fit(X)
    assert enc._to_extract == expected_to_extract
    assert enc.features_per_column_ == expected_features_per_column_

    X = get_date_array()
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_to_extract = ["year", "month", "day", "hour", "dayofweek"]
    expected_features_per_column_ = {
        0: ["year", "month", "day", "dayofweek"],
        1: ["month", "day", "dayofweek"],
        2: ["year", "month", "day", "dayofweek"],
    }
    enc.fit(X)
    assert enc._to_extract == expected_to_extract
    assert enc.features_per_column_ == expected_features_per_column_

    # Datetimes
    X = get_datetime_array()
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_to_extract = ["year", "month", "day", "hour", "dayofweek"]
    expected_features_per_column_ = {
        0: ["year", "month", "day", "hour", "dayofweek", "total_time"],
        1: ["month", "day", "hour", "dayofweek", "total_time"],
        2: ["year", "month", "day", "hour", "dayofweek"],
    }
    enc.fit(X)
    assert enc._to_extract == expected_to_extract
    assert enc.features_per_column_ == expected_features_per_column_

    X = get_datetime_array()
    enc = DatetimeEncoder(extract_until="minute")
    expected_to_extract = ["year", "month", "day", "hour", "minute"]
    expected_features_per_column_ = {
        0: ["year", "month", "day", "hour", "minute", "total_time"],
        1: ["month", "day", "hour", "minute"],
        2: ["year", "month", "day", "hour"],
    }
    enc.fit(X)
    assert enc._to_extract == expected_to_extract
    assert enc.features_per_column_ == expected_features_per_column_

    # Dirty Datetimes
    X = get_dirty_datetime_array()
    enc = DatetimeEncoder()
    expected_to_extract = ["year", "month", "day", "hour"]
    expected_features_per_column_ = {
        0: ["year", "month", "day", "hour", "total_time"],
        1: ["month", "day", "hour", "total_time"],
        2: ["year", "month", "day", "hour"],
    }
    enc.fit(X)
    assert enc._to_extract == expected_to_extract
    assert enc.features_per_column_ == expected_features_per_column_

    # Datetimes with TZ
    X = get_datetime_with_TZ_array()
    enc = DatetimeEncoder()
    expected_to_extract = ["year", "month", "day", "hour"]
    expected_features_per_column_ = {0: ["year", "month", "day", "hour", "total_time"]}
    enc.fit(X)
    assert enc._to_extract == expected_to_extract
    assert enc.features_per_column_ == expected_features_per_column_

    # Feature names
    # Without column names
    X = get_datetime_array()
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_feature_names = [
        "0_year",
        "0_month",
        "0_day",
        "0_hour",
        "0_dayofweek",
        "0_total_time",
        "1_month",
        "1_day",
        "1_hour",
        "1_dayofweek",
        "1_total_time",
        "2_year",
        "2_month",
        "2_day",
        "2_hour",
        "2_dayofweek",
    ]
    enc.fit(X)
    assert enc.get_feature_names_out() == expected_feature_names

    # With column names
    X = get_datetime_array()
    X = pd.DataFrame(X)
    X.columns = ["col1", "col2", "col3"]
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_feature_names = [
        "col1_year",
        "col1_month",
        "col1_day",
        "col1_hour",
        "col1_dayofweek",
        "col1_total_time",
        "col2_month",
        "col2_day",
        "col2_hour",
        "col2_dayofweek",
        "col2_total_time",
        "col3_year",
        "col3_month",
        "col3_day",
        "col3_hour",
        "col3_dayofweek",
    ]
    enc.fit(X)
    assert enc.get_feature_names_out() == expected_feature_names


def test_transform() -> None:
    # Dates
    X = get_date_array()
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_result = np.array(
        [
            [2020, 1, 1, 2, 1, 2, 3, 2020, 1, 3, 4],
            [2021, 2, 3, 2, 2, 4, 1, 2021, 2, 5, 4],
            [2022, 1, 1, 5, 12, 25, 4, 2022, 1, 3, 0],
            [2023, 2, 3, 4, 2, 4, 1, 2023, 2, 5, 6],
        ]
    )
    enc.fit(X)
    assert np.allclose(enc.transform(X), expected_result, equal_nan=True)

    enc = DatetimeEncoder(add_day_of_the_week=False)
    expected_result = np.array(
        [
            [2020, 1, 1, 1, 2, 2020, 1, 3],
            [2021, 2, 3, 2, 4, 2021, 2, 5],
            [2022, 1, 1, 12, 25, 2022, 1, 3],
            [2023, 2, 3, 2, 4, 2023, 2, 5],
        ]
    )
    enc.fit(X)
    assert np.allclose(enc.transform(X), expected_result, equal_nan=True)

    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_result = np.array(
        [
            [2020, 1, 1, 2, 1, 2, 3, 2020, 1, 3, 4],
            [2021, 2, 3, 2, 2, 4, 1, 2021, 2, 5, 4],
            [2022, 1, 1, 5, 12, 25, 4, 2022, 1, 3, 0],
            [2023, 2, 3, 4, 2, 4, 1, 2023, 2, 5, 6],
        ]
    )
    enc.fit(X)
    assert np.allclose(enc.transform(X), expected_result, equal_nan=True)

    # Datetimes
    X = get_datetime_array()[:, 0].reshape(-1, 1)
    enc = DatetimeEncoder(add_day_of_the_week=True)
    # Check that the "total_time" feature is working
    expected_result = np.array(
        [
            [2020, 1, 1, 10, 2, 0],
            [2021, 2, 3, 12, 2, 0],
            [2022, 1, 1, 23, 5, 0],
            [2023, 2, 3, 11, 4, 0],
        ]
    ).astype(np.float64)
    # Time from epochs in seconds
    expected_result[:, 5] = (X.astype("int64") // 1e9).astype(np.float64).reshape(-1)

    enc.fit(X)
    X_trans = enc.transform(X)
    assert np.allclose(X_trans, expected_result, equal_nan=True)

    # Check if we find back the date from the time to epoch
    assert (
        (
            pd.to_datetime(X_trans[:, 5], unit="s") - pd.to_datetime(X.reshape(-1))
        ).total_seconds()
        == 0
    ).all()

    # Dirty datetimes
    X = get_dirty_datetime_array()[:, 0].reshape(-1, 1)
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_result = np.array(
        [
            [2020, 1, 1, 10, 2, 0],
            [np.nan] * 6,
            [2022, 1, 1, 23, 5, 0],
            [2023, 2, 3, 11, 4, 0],
        ]
    )
    # Time from epochs in seconds
    expected_result[:, 5] = (X.astype("int64") // 1e9).astype(np.float64).reshape(-1)
    expected_result[1, 5] = np.nan
    enc.fit(X)
    X_trans = enc.transform(X)
    assert np.allclose(X_trans, expected_result, equal_nan=True)

    # Datetimes with TZ
    # If the dates are timezone-aware, all the feature extractions should
    # be done in the provided timezone.
    # But the full time to epoch should correspond to the true number of
    # seconds between epoch time and the time of the date.
    X = get_datetime_with_TZ_array()
    enc = DatetimeEncoder(add_day_of_the_week=True)
    expected_result = np.array(
        [
            [2020, 1, 1, 10, 2, 0],
            [2021, 2, 3, 12, 2, 0],
            [2022, 1, 1, 23, 5, 0],
            [2023, 2, 3, 11, 4, 0],
        ]
    ).astype(np.float64)
    # Time from epochs in seconds
    expected_result[:, 5] = (
        (X.iloc[:, 0].view(dtype="int64") // 1e9)
        .astype(np.float64)
        .to_numpy()
        .reshape(-1)
    )
    enc.fit(X)
    X_trans = enc.transform(X)
    assert np.allclose(X_trans, expected_result, equal_nan=True)

    # Check if we find back the date from the time to epoch
    assert (
        (
            pd.to_datetime(X_trans[:, 5], unit="s")
            .tz_localize("utc")
            .tz_convert(X.iloc[:, 0][0].tz)
            - pd.DatetimeIndex(X.iloc[:, 0])
        ).total_seconds()
        == 0
    ).all()

    # Check if it's working when the date is constant
    X = get_constant_date_array()
    enc = DatetimeEncoder(add_day_of_the_week=True)
    assert enc.fit_transform(X).shape[1] == 0


def test_check_fitted_datetime_encoder() -> None:
    """Test that calling transform before fit raises an error"""
    X = get_datetime_array()[:, 0].reshape(-1, 1)
    enc = DatetimeEncoder(add_day_of_the_week=True)
    with pytest.raises(NotFittedError):
        enc.transform(X)

    # Check that it works after fit
    enc.fit(X)
    enc.transform(X)
