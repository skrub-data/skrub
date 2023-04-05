import collections
import warnings
from typing import Any, Hashable

import numpy as np
import pandas as pd
from pandas._libs.tslibs.parsing import guess_datetime_format
from sklearn.utils import check_array

try:
    # Works for sklearn >= 1.0
    from sklearn.utils import parse_version  # noqa
except ImportError:
    # Works for sklearn < 1.0
    from sklearn.utils.fixes import _parse_version as parse_version  # noqa


class LRUDict:
    """dict with limited capacity

    Using LRU eviction avoids memorizing a full dataset"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def __getitem__(self, key: Hashable):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return -1

    def __setitem__(self, key: Hashable, value: Any):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def __contains__(self, key: Hashable):
        return key in self.cache


def check_input(X) -> np.ndarray:
    """
    Check input with sklearn standards.
    Also converts X to a numpy array if not already.
    """
    # TODO check for weird type of input to pass scikit learn tests
    #  without messing with the original type too much

    X_ = check_array(
        X,
        dtype=None,
        ensure_2d=True,
        force_all_finite=False,
    )
    # If the array contains both NaNs and strings, convert to object type
    if X_.dtype.kind in {"U", "S"}:  # contains strings
        if np.any(X_ == "nan"):  # missing value converted to string
            return check_array(
                np.array(X, dtype=object),
                dtype=None,
                ensure_2d=True,
                force_all_finite=False,
            )

    return X_


def _infer_date_format(date_column, n_trials=100) -> str:
    """Infer the date format of a date column,
    by finding a format which should work for all dates in the column.

    Parameters
    ----------
    date_column : pandas.Series
        A column of dates, as strings.
    n_trials : int, default=100
        Number of rows to use to infer the date format.

    Returns
    -------
    date_format : str
        The date format inferred from the column.
        If no format could be inferred, returns None.
    """
    # shuffle the column to avoid bias
    date_column_shuffled = date_column.sample(frac=1, random_state=42)
    # remove nan values
    date_column_shuffled = date_column_shuffled.dropna()
    # TODO for speed, we could filter for rows which have an
    # higher chance of resolving ambiguity (with number greater than 12)
    # select the first n_trials rows
    date_column_sample = date_column_shuffled.iloc[:n_trials]
    # try to infer the date format
    date_format = date_column_sample.apply(lambda x: guess_datetime_format(x))
    # if one format is None, return None
    if date_format.isnull().any() or not len(date_format):
        return None
    elif date_format.nunique() == 1:
        # one format works for all the rows
        # check if another format works for all the rows
        # if so, raise a warning
        date_format_dayfirst = date_column_sample.apply(
            lambda x: guess_datetime_format(x, dayfirst=True),
        )
        # if it worked for all the rows, date_format.nunique() == 1
        if date_format.nunique() == 1:
            warnings.warn(
                f"""
                Both {date_format[0]} and {date_format_dayfirst[0]} are valid
                formats for the dates in column {date_column.name}.
                Format {date_format[0]} will be used.
                """,
                UserWarning,
                stacklevel=2,
            )
        return date_format[0]
    elif date_format.nunique() > 2:
        return None  # TODO: handle this case?
    # otherwise, find if one of the two
    # format works for all the rows
    else:
        date_format = date_format.dropna().unique()
        first_format_works = False
        second_format_works = False
        try:
            pd.to_datetime(date_column_sample, format=date_format[0], errors="raise")
            first_format_works = True
        except ValueError:
            pass
        try:
            pd.to_datetime(date_column_sample, format=date_format[1], errors="raise")
            second_format_works = True
        except ValueError:
            pass
        if first_format_works and second_format_works:
            warnings.warn(
                f"""
                Both {date_format[0]} and {date_format[1]} are valid
                formats for the dates in column {date_column.name}.
                Format {date_format[0]} will be used.
                """,
                UserWarning,
                stacklevel=2,
            )
            return date_format[0]
        elif first_format_works:
            return date_format[0]
        elif second_format_works:
            return date_format[1]
        else:
            return None
