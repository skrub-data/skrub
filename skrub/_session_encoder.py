"""
The SessionEncoder is a transformer that takes as input:
- a "timestamp" column, which identifies the time of an event
- a "by" column or list of columns, which identifies a user
- a "session_gap" value, which identifies the maximum allowed gap between events
in a session

It returns a dataframe with the same number of rows as the input, but with the
column "session_id": a unique identifier for each session, which is a combination
of the "by" column(s) and a session number
"""

import numbers
from collections.abc import Iterable

import pandas as pd
from packaging.version import parse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._dispatch import dispatch, raise_dispatch_unregistered_type
from ._utils import random_string

try:
    import polars as pl
except ImportError:
    pass


@dispatch
def _check_is_new_session(X, group_by, timestamp, session_gap):
    raise_dispatch_unregistered_type(X, kind="Dataframe")


@_check_is_new_session.specialize("pandas")
def _check_is_new_session_pandas(X, group_by, timestamp, session_gap):
    # pandas 3.0 changed the resolution of astype(int) for datetime columns from
    # nanoseconds to milliseconds, so we need to adjust the time difference calculation
    # accordingly
    if parse(pd.__version__).major <= 2:
        # check if the time difference between events exceeds the session gap
        time_diff = (
            X[timestamp].astype(int).diff().fillna(0) // 10**6 > session_gap * 60 * 1000
        )
    else:
        time_diff = (
            X[timestamp].astype(int).diff().fillna(0) // 10**3 > session_gap * 60 * 1000
        )
    if not group_by:
        return time_diff
    # check if the "group_by" column changes
    char_diff = (X[group_by].diff().fillna(0) > 0).any(axis=1)
    # a new session starts if either the "group_by" column changes or the time gap is
    # exceeded
    is_new_session = char_diff | time_diff
    return is_new_session


@_check_is_new_session.specialize("polars")
def _check_is_new_session_polars(X, group_by, timestamp, session_gap):
    # check if the time difference between events exceeds the session gap
    time_diff = (
        X[timestamp].dt.epoch("ms").diff().fill_null(0) > session_gap * 60 * 1000
    )
    if not group_by:
        return time_diff
    # check if the "group_by" column changes
    char_diff = X.select(
        pl.any_horizontal(pl.col(group_by).diff().fill_null(0) > 0)
    ).to_series()
    # a new session starts if either the "group_by" column changes or the time gap is
    # exceeded
    is_new_session = char_diff | time_diff
    return is_new_session


@dispatch
def _add_session_id(X, is_new_session, column_name):
    raise_dispatch_unregistered_type(X, kind="Dataframe")


@_add_session_id.specialize("pandas")
def _add_session_id_pandas(X, is_new_session, column_name):
    # Compute cumulative sum of is_new_session to create session IDs
    X[column_name] = is_new_session.cumsum()
    return X


@_add_session_id.specialize("polars")
def _add_session_id_polars(X, is_new_session, column_name):
    # Add session_id by computing cumulative sum of is_new_session
    return X.with_columns(is_new_session.cum_sum().alias(column_name))


@dispatch
def _factorize_column(X, column_name):
    raise_dispatch_unregistered_type(X, kind="Dataframe")


@_factorize_column.specialize("pandas")
def _factorize_column_pandas(X, column_name):
    if sbd.is_numeric(X[column_name]):
        return X[column_name]
    codes, _ = pd.factorize(X[column_name])
    return codes


@_factorize_column.specialize("polars")
def _factorize_column_polars(X, column_name):
    import polars as pl

    if sbd.is_numeric(X[column_name]):
        return X[column_name]
    return X[column_name].cast(pl.Categorical).to_physical()


class SessionEncoder(TransformerMixin, BaseEstimator):
    """Encode sessions from a dataframe.

    A session is defined as a sequence of events  where consecutive events are separated
    by at most ``session_gap`` minutes. Additionally, it is possible to provide a column
    or list of columns that identifies the user (specified by the ``group_by`` column).
    When the time gap between consecutive events exceeds ``session_gap``, or
    when the user changes, a new session begins. All unrelated columns are passed
    through unchanged.

    Parameters
    ----------
    timestamp_col : str
        The name of the column that identifies the time of an event. This column
        is used to determine the start and end of a session.

    group_by : optional[str, list[str]], default=None
        The name of the column, or list of columns, to group by. This parameter
        is used to group events into sessions by, for example, user. If not
        provided, sessions are detected based on the time gap between events, and all
        events are considered to belong to the same user (or group).

    session_gap : int, default=30
        The maximum gap (in minutes) between events in a session. If the gap
        between two events exceeds this value, they are considered to be in
        different sessions.

    Attributes
    ----------
    all_inputs_ : list of str
        All column names in the input dataframe.

    all_outputs_: list of str
        All column names in the input dataframe plus the new column that identifies
        the session, with name "{timestamp}_session_id".

    Examples
    --------
    Consider this example where we have a dataframe with user events, and we want
    to identify sessions based on a 30-minute gap between events for each user:

    >>> import pandas as pd
    >>> from datetime import datetime, timedelta
    >>> encoder = SessionEncoder(
    ...     group_by='user_id', timestamp_col='timestamp', session_gap=30
    ... )

    >>> # Create a sample dataframe with events from different users
    >>> data = {
    ...     'user_id': ['alice', 'alice', 'alice', 'bob', 'bob'],
    ...     'timestamp': [
    ...         pd.Timestamp('2024-01-01 10:00:00'),
    ...         pd.Timestamp('2024-01-01 10:05:00'),  # 5 min later, same session
    ...         pd.Timestamp('2024-01-01 11:00:00'),  # 55 min later, new session
    ...         pd.Timestamp('2024-01-01 10:00:00'),  # Different user
    ...         pd.Timestamp('2024-01-01 10:20:00'),  # 20 min later, same session
    ...     ],
    ...     'action': ['login', 'view', 'logout', 'login', 'purchase']
    ... }
    >>> df = pd.DataFrame(data)
    >>> df
        user_id           timestamp   action
    0    alice 2024-01-01 10:00:00    login
    1    alice 2024-01-01 10:05:00     view
    2    alice 2024-01-01 11:00:00   logout
    3      bob 2024-01-01 10:00:00    login
    4      bob 2024-01-01 10:20:00 purchase

    >>> result = encoder.fit_transform(df)
    >>> result
       user_id           timestamp   action  timestamp_session_id
    0    alice 2024-01-01 10:00:00    login                     0
    1    alice 2024-01-01 10:05:00     view                     0
    2    alice 2024-01-01 11:00:00 purchase                     1
    3      bob 2024-01-01 10:00:00    login                     2
    4      bob 2024-01-01 10:20:00 purchase                     2

    In this example:

    - Alice's first two events (10:00 and 10:05) are 5 minutes apart, so they form
      session 0.
    - Alice's third event (11:00) is 55 minutes after the previous one, exceeding
      the 30-minute gap, so it forms a new session (session 1).
    - Bob's events form session 2 (different user), with both events within the
      30-minute window.

    You can also identify users by multiple columns. For instance, the same user
    on different devices should have separate sessions:

    >>> encoder_multi = SessionEncoder(
    ...     group_by=['user_id', 'device_id'],
    ...     timestamp_col='timestamp',
    ...     session_gap=30
    ... )

    >>> # Create a sample dataframe where user_id + device_id identifies a user
    >>> data_multi = {
    ...     'user_id': [1, 1, 1, 1, 2, 2],
    ...     'device_id': ['mobile', 'mobile', 'desktop', 'desktop', 'mobile', 'mobile'],
    ...     'timestamp': [
    ...         pd.Timestamp('2024-01-01 10:00:00'),
    ...         pd.Timestamp('2024-01-01 10:10:00'),  # 10 min later, same session
    ...         pd.Timestamp('2024-01-01 10:05:00'),  # Different device (sorted),
    ...                                                 # different session
    ...         pd.Timestamp('2024-01-01 10:20:00'),  # 15 min later, same session
    ...         pd.Timestamp('2024-01-01 10:00:00'),  # Different user
    ...         pd.Timestamp('2024-01-01 10:15:00'),  # 15 min later, same session
    ...     ],
    ...     'action': ['view', 'purchase', 'view', 'checkout', 'login', 'view']
    ... }
    >>> df_multi = pd.DataFrame(data_multi)
    >>> result_multi = encoder_multi.fit_transform(df_multi)
    >>> result_multi
       user_id  device_id           timestamp     action  timestamp_session_id
    0        1    desktop 2024-01-01 10:05:00       view                     0
    1        1    desktop 2024-01-01 10:20:00   checkout                     0
    2        1     mobile 2024-01-01 10:00:00       view                     1
    3        1     mobile 2024-01-01 10:10:00   purchase                     1
    4        2     mobile 2024-01-01 10:00:00      login                     2
    5        2     mobile 2024-01-01 10:15:00       view                     2

    In this example:

    - User 1 on "desktop" has session 0.
    - User 1 on "mobile" has session 1 (different device, so separate session).
    - User 2 on "mobile" has session 2 (different user).

    You can also use SessionEncoder without a user identifier column. In this case,
    sessions are separated only by time gaps. This is useful for analyzing a single
    timeseries or events that don't have a user dimension:

    >>> encoder_no_group = SessionEncoder(
    ...     group_by=None,
    ...     timestamp_col='timestamp',
    ...     session_gap=30
    ... )

    >>> # Create a sample dataframe with only timestamps
    >>> data_no_group = {
    ...     'timestamp': [
    ...         pd.Timestamp('2024-01-01 10:00:00'),
    ...         pd.Timestamp('2024-01-01 10:10:00'),  # 10 min gap
    ...         pd.Timestamp('2024-01-01 10:15:00'),  # 5 min gap, still in session
    ...         pd.Timestamp('2024-01-01 11:00:00'),  # 45 min gap, new session
    ...         pd.Timestamp('2024-01-01 11:10:00'),  # 10 min gap, continue session
    ...     ],
    ...     'event_type': ['start', 'action', 'action', 'restart', 'action']
    ... }
    >>> df_no_group = pd.DataFrame(data_no_group)
    >>> result_no_group = encoder_no_group.fit_transform(df_no_group)
    >>> result_no_group
                 timestamp event_type  timestamp_session_id
    0 2024-01-01 10:00:00      start                     0
    1 2024-01-01 10:10:00     action                     0
    2 2024-01-01 10:15:00     action                     0
    3 2024-01-01 11:00:00    restart                     1
    4 2024-01-01 11:10:00     action                     1

    In this example:

    - Events at 10:00, 10:10, and 10:15 form session 0 (all gaps < 30 min).
    - The event at 11:00 starts a new session 1 (45 min gap > 30 min).
    - The event at 11:10 continues session 1 (10 min gap < 30 min).


    """

    def __init__(self, timestamp_col, group_by=None, session_gap=30):
        self.timestamp_col = timestamp_col
        self.group_by = group_by
        self.session_gap = session_gap

    def fit(self, X, y=None):
        """Fit the transformer to the data.

        Parameters
        ----------
        X : pandas.DataFrame or polars.DataFrame
            The input dataframe.

        y : None
            Ignored.

        Returns
        -------
        self : SessionEncoder
            The fitted transformer.
        """
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        """Fit the transformer to the data and return the transformed dataframe.

        Parameters
        ----------
        X : pandas.DataFrame or polars.DataFrame
            The input dataframe.

        y : None
            Ignored.

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            The transformed dataframe with session information.
        """
        self.all_inputs_ = sbd.column_names(X)
        # check that the required columns are present in the input dataframe
        if self.group_by is not None:
            if isinstance(self.group_by, str):
                self.group_by_columns = [self.group_by]
            elif isinstance(self.group_by, Iterable) and not isinstance(
                self.group_by, str
            ):
                self.group_by_columns = list(self.group_by)
            else:
                raise TypeError("group_by must be a string, a list of strings, or None")
        if self.group_by is not None:
            for col in self.group_by_columns:
                if col not in self.all_inputs_:
                    raise ValueError(f"Column '{col}' not found in input dataframe")

        if self.timestamp_col not in self.all_inputs_:
            raise ValueError(
                f"Column '{self.timestamp_col}' not found in input dataframe"
            )

        # check the correctness of the values of session_gap
        if not isinstance(self.session_gap, numbers.Number) or self.session_gap <= 0:
            raise ValueError("session_gap must be a positive number")

        # sort the input dataframe by the "group_by" and "timestamp" columns
        sort_by = (
            self.group_by_columns + [self.timestamp_col]
            if self.group_by is not None
            else [self.timestamp_col]
        )
        X_sorted = sbd.sort(X, by=sort_by)

        X_factorized, factorized_by = self._factorize_columns(X_sorted)
        # mark the start of a new session by checking the difference
        is_new_session = _check_is_new_session(
            X_factorized, factorized_by, self.timestamp_col, self.session_gap
        )
        # add the session id
        session_col_name = f"{self.timestamp_col}_session_id"
        X_with_session_id = _add_session_id(
            X_factorized, is_new_session, session_col_name
        )

        # drop the factorized "group_by" column if the original "group_by"
        # column was not numeric
        to_drop = [col for col in factorized_by if col not in self.group_by_columns]
        X_with_session_id = sbd.drop_columns(X_with_session_id, to_drop)

        self.all_outputs_ = sbd.column_names(X_with_session_id)
        return X_with_session_id

    def transform(self, X):
        """Transform the data by encoding sessions.

        Parameters
        ----------
        X : pandas.DataFrame or polars.DataFrame
            The input dataframe.

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            The transformed dataframe with session information.
        """
        check_is_fitted(self)
        return self.fit_transform(X)

    def _factorize_columns(self, X):
        # convert group_by column to string if it's not already, to ensure
        # that the diff operation works correctly
        if not self.group_by:
            return X, []
        factorized_columns = {
            f"{col}_factorized_skrub_{random_string()}": _factorize_column(X, col)
            if not sbd.is_numeric(X[col])
            else X[col]
            for col in self.group_by_columns
        }

        X_factorized = sbd.with_columns(X, **factorized_columns)

        return X_factorized, list(factorized_columns.keys())

    def get_feature_names_out(self, input_features=None):
        """Return the column names of the output of ``transform`` as a list of strings.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Ignored.

        Returns
        -------
        list of strings
            The column names.
        """
        check_is_fitted(self)
        return self.all_outputs_
