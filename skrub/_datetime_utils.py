import warnings

import numpy as np
import pandas as pd
from pandas._libs.tslibs.parsing import (
    guess_datetime_format as pd_guess_datetime_format,
)
from sklearn.utils.fixes import parse_version
from sklearn.utils.validation import check_random_state


def _mixed_format():
    pandas_version = pd.__version__
    if parse_version("2.0.0") < parse_version(pandas_version):
        return "mixed"
    return None


def is_column_datetime_parsable(column):
    """Check whether a 1d array can be converted into a \
    :class:`pandas.DatetimeIndex`.

    Parameters
    ----------
    column : array-like of shape ``(n_samples,)``

    Returns
    -------
    is_dt_parsable : bool
    """
    # The aim of this section is to remove any columns of int, float or bool
    # casted as object.
    # Pandas < 2.0.0 raise a deprecation warning instead of an error.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        try:
            if np.array_equal(column, column.astype(np.float64), equal_nan=True):
                return False
        except (ValueError, TypeError):
            pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            # format=mixed parses entries individually,
            # avoiding ValueError when both date and datetime formats
            # are present.
            # At this stage, the format itself doesn't matter.
            _ = pd.to_datetime(column, format=_mixed_format())
            return True
        except (pd.errors.ParserError, ValueError, TypeError):
            return False


def guess_datetime_format(column, random_state=None):
    """Infer the format of a 1d array.

    This functions uses Pandas ``guess_datetime_format`` routine for both
    dayfirst and monthfirst case, and select either format when using one
    give a unify format on the array.

    When both dayfirst and monthfirst format are possible, we select
    monthfirst by default.

    You can overwrite this behaviour by setting the format argument of the
    caller function.
    Setting a format always take precedence over infering it using
    ``_guess_datetime_format``.
    """
    # Subsample samples for fast format estimation
    column = pd.Series(column).dropna().to_numpy()
    n_samples = 30
    size = min(column.shape[0], n_samples)
    rng = check_random_state(random_state)
    column = rng.choice(column, size=size, replace=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        # In Pandas 2.1, pd.to_datetime has a different behavior
        # when the items of column are np.str_ instead of str.
        # TODO: find the bug and remove this line
        column = pd.Series(map(str, column))

        # pd.unique handles None
        month_first_formats = column.apply(
            pd_guess_datetime_format, dayfirst=False
        ).unique()
        day_first_formats = column.apply(
            pd_guess_datetime_format, dayfirst=True
        ).unique()

    if len(month_first_formats) == 1 and month_first_formats[0] is not None:
        return str(month_first_formats[0])

    elif len(day_first_formats) == 1 and day_first_formats[0] is not None:
        return str(day_first_formats[0])

    return None
