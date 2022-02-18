import pandas as pd
from dirty_cat.datetime_encoder import DatetimeEncoder
import numpy as np


def get_datetimes_array():
    return np.array([pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                  pd.to_datetime(["2021-02-03", "2020-02-04", "2021-02-05"]),
                  pd.to_datetime(["2022-01-01", "2020-01-02", "2022-01-03"]),
                  pd.to_datetime(["2023-02-03", "2020-02-04", "2023-02-05"])])

def test_fit():
    X = np.array([pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                  pd.to_datetime(["2021-02-03", "2020-02-04", "2021-02-05"]),
                  pd.to_datetime(["2022-01-01", "2020-01-02", "2022-01-03"]),
                  pd.to_datetime(["2023-02-03", "2020-02-04", "2023-02-05"])])

    expected_to_extract_full = ["year", "month", "day", "hour", "minute"]
    expected_to_extract = {0: ["year", "month", "day", "dayofweek"],
                           1: ["month", "day", "dayofweek"],
                           2: ["year", "month", "day", "dayofweek"]}

    enc = DatetimeEncoder()
    enc.fit(X)

    assert enc.to_extract_full == expected_to_extract_full
    assert enc.to_extract == expected_to_extract


def test_transform():
    X = get_datetimes_array()
    expected_result = np.array([[2020, 1, 1, 2, 1, 2, 3, 2020, 1, 3, 4],
                                [2021, 2, 3, 2, 2, 4, 1, 2021, 2, 5, 4],
                                [2022, 1, 1, 5, 1, 2, 3, 2022, 1, 3, 0],
                                [2023, 2, 3, 4, 2, 4, 1, 2023, 2, 5, 6]])

    enc = DatetimeEncoder()
    enc.fit(X)
    assert (enc.transform(X) == expected_result).all()
