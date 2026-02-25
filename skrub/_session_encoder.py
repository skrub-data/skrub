"""
The SessionEncoder is a transformer that takes as input:
- a "by" column, which identifies a user
- a "timestamp" column, which identifies the time of an event
- a "session_gap" value, which identifies the maximum allowed gap between events
in a session

It returns a dataframe with the same number of rows as the input, but with the
column "session_id": a unique identifier for each session, which is a combination
of the "by" column and a session number
"""

import numbers

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._dispatch import dispatch


@dispatch
def _check_is_new_session(X, by, timestamp, session_gap):
    # Avoid circular import
    from ._dispatch import raise_dispatch_unregistered_type

    raise_dispatch_unregistered_type(X, kind="Dataframe")


@_check_is_new_session.specialize("pandas")
def _check_is_new_session_pandas(X, by, timestamp, session_gap):
    # check if the "by" column changes
    char_diff = X[by].diff().fillna(0) > 0
    # check if the time difference between events exceeds the session gap
    time_diff = (
        X[timestamp].astype(int).diff().fillna(0) // 10**3 > session_gap * 60 * 1000
    )
    # a new session starts if either the "by" column changes or the time gap is
    # exceeded
    is_new_session = char_diff | time_diff
    return is_new_session


@_check_is_new_session.specialize("polars")
def _check_is_new_session_polars(X, by, timestamp, session_gap):
    # check if the "by" column changes
    char_diff = X[by].diff().fill_null(0) > 0
    # check if the time difference between events exceeds the session gap
    time_diff = (
        X[timestamp].dt.epoch("ms").diff().fill_null(0) > session_gap * 60 * 1000
    )
    # a new session starts if either the "by" column changes or the time gap is
    # exceeded
    is_new_session = char_diff | time_diff
    return is_new_session


@dispatch
def _factorize_column(X, column_name):
    # Avoid circular import
    from ._dispatch import raise_dispatch_unregistered_type

    raise_dispatch_unregistered_type(X, kind="Dataframe")


@_factorize_column.specialize("pandas")
def _factorize_column_pandas(X, column_name):
    codes, _ = pd.factorize(X[column_name])
    return codes


@_factorize_column.specialize("polars")
def _factorize_column_polars(X, column_name):
    import polars as pl

    # TODO: update this according to the proper polars API
    return X[column_name].cast(pl.Categorical).to_physical()


@dispatch
def _add_session_id(X, is_new_session):
    # Avoid circular import
    from ._dispatch import raise_dispatch_unregistered_type

    raise_dispatch_unregistered_type(X, kind="Dataframe")


@_add_session_id.specialize("pandas")
def _add_session_id_pandas(X, is_new_session):
    # Compute cumulative sum of is_new_session to create session IDs
    X_copy = X.copy()
    X_copy["session_id"] = is_new_session.cumsum()
    return X_copy


@_add_session_id.specialize("polars")
def _add_session_id_polars(X, is_new_session):
    # Add session_id by computing cumulative sum of is_new_session
    return X.with_columns(is_new_session.cum_sum().alias("session_id"))


class SessionEncoder(TransformerMixin, BaseEstimator):
    """Encode sessions from a dataframe.

    A session is defined as a sequence of events from the same user (specified by
    the `by` column) where consecutive events are separated by at most `session_gap`
    minutes. When the time gap between consecutive events exceeds `session_gap`, or
    when the user changes, a new session begins.

    Parameters
    ----------
    by : str
        The name of the column that identifies a user. This column is used to
        group events into sessions.

    timestamp : str
        The name of the column that identifies the time of an event. This column
        is used to determine the start and end of a session.

    session_gap : int, default=30
        The maximum gap (in minutes) between events in a session. If the gap
        between two events exceeds this value, they are considered to be in
        different sessions.

    Attributes
    ----------
    all_inputs_ : list of str
        All column names in the input dataframe.

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime, timedelta
    >>> encoder = SessionEncoder(by='user_id', timestamp='timestamp', session_gap=30)

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
       user_id           timestamp   action  session_id
    0    alice 2024-01-01 10:00:00    login           0
    1    alice 2024-01-01 10:05:00     view           0
    2    alice 2024-01-01 11:00:00   logout           1
    3      bob 2024-01-01 10:00:00    login           2
    4      bob 2024-01-01 10:20:00 purchase           2

    In this example:
    - Alice's first two events (10:00 and 10:05) are 5 minutes apart, so they form
      session 1.
    - Alice's third event (11:00) is 55 minutes after the previous one, exceeding
      the 30-minute gap, so it forms a new session (session 2).
    - Bob's events form session 3 (different user), with both events within the
      30-minute window.
    """

    def __init__(self, by, timestamp, session_gap=30):
        self.by = by
        self.timestamp = timestamp
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
        if self.by not in self.all_inputs_:
            raise ValueError(f"Column '{self.by}' not found in input dataframe")
        if self.timestamp not in self.all_inputs_:
            raise ValueError(f"Column '{self.timestamp}' not found in input dataframe")

        # check the correctness of the values of session_gap
        if not isinstance(self.session_gap, numbers.Number) or self.session_gap <= 0:
            raise ValueError("session_gap must be a positive number")

        # sort the input dataframe by the "by" and "timestamp" columns
        X_sorted = sbd.sort(X, by=[self.by, self.timestamp])  # noqa

        # convert by column to string if it's not already, to ensure
        # that the diff operation works correctly
        if not sbd.is_numeric(X_sorted[self.by]):
            factorized_by = f"{self.by}_factorized"
            X_factorized = sbd.with_columns(
                X_sorted, **{factorized_by: _factorize_column(X_sorted, self.by)}
            )
        else:
            factorized_by = self.by
            X_factorized = X_sorted
        # mark the start of a new session by checking the difference
        is_new_session = _check_is_new_session(
            X_factorized, factorized_by, self.timestamp, self.session_gap
        )
        # add the session id
        X_with_session_id = _add_session_id(X_factorized, is_new_session)

        # drop the factorized "by" column if the original "by" column was not numeric
        if factorized_by != self.by:
            X_with_session_id = sbd.drop_columns(
                X_with_session_id, columns=[factorized_by]
            )

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
        check_is_fitted()
        return self.fit_transform(X)
