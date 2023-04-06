import collections
import warnings
from typing import Any, Hashable, Optional

import numpy as np
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


def _infer_date_format(date_column: pd.Series, n_trials: int = 100) -> Optional[str]:
    """Infer the date format of a date column,
    by finding a format which should work for all dates in the column.

    Parameters
    ----------
    date_column : :class:`~pandas.Series`
        A column of dates, as strings.
    n_trials : int, default=100
        Number of rows to use to infer the date format.

    Returns
    -------
    Optional[str]
        The date format inferred from the column.
        If no format could be inferred, returns None.
    """
    if len(date_column) == 0:
        return
    date_column_sample = date_column.dropna().sample(
        frac=min(n_trials / len(date_column), 1), random_state=42
    )
    # try to infer the date format
    # see if either dayfirst or monthfirst works for all the rows
    with warnings.catch_warnings():
        # pandas warns when dayfirst is not strictly applied
        warnings.simplefilter("ignore")
        date_format_monthfirst = date_column_sample.apply(
            lambda x: guess_datetime_format(x)
        )
        date_format_dayfirst = date_column_sample.apply(
            lambda x: guess_datetime_format(x, dayfirst=True),
        )
    # if one row could not be parsed, return None
    if date_format_monthfirst.isnull().any() or date_format_dayfirst.isnull().any():
        return
    # even with dayfirst=True, monthfirst format can be inferred
    # so we need to check if the format is the same for all the rows
    elif date_format_monthfirst.nunique() == 1:
        # one monthfirst format works for all the rows
        # check if another format works for all the rows
        # if so, raise a warning
        if date_format_dayfirst.nunique() == 1:
            warnings.warn(
                f"""
                Both {date_format_monthfirst[0]} and {date_format_dayfirst[0]} are valid
                formats for the dates in column {date_column.name}.
                Format {date_format_monthfirst[0]} will be used.
                """,
                UserWarning,
                stacklevel=2,
            )
        return date_format_monthfirst[0]
    elif date_format_dayfirst.nunique() == 1:
        # only this format works for all the rows
        return date_format_dayfirst[0]
    else:
        # more than two different formats were found
        # TODO: maybe we could deal with this case
        return
