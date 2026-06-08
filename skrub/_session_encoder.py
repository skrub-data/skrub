"""
The SessionEncoder is a transformer that takes as input:
- a "timestamp" column, which identifies the time of an event
- a "split_by" column or list of columns, which identifies a user
- a "session_gap" value, which identifies the maximum allowed gap in seconds
  between events in a session

It returns a dataframe with the same number of rows as the input, but with an
additional column that identifies the session to which each event belongs.
The name of the session column is "{timestamp}_{suffix}", where "timestamp" is the name
of the timestamp column, and "suffix" is a string that can be set via the "suffix"
parameter (default is "session_id"). The session column contains a unique identifier for
each session, which is a combination of the "split_by" column(s) and a session number
"""

import numbers
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from . import selectors as s
from ._dispatch import dispatch, raise_dispatch_unregistered_type
from ._utils import random_string

try:
    import polars as pl
except ImportError:
    pass


@dispatch
def _add_session_column(X, split_by, timestamp_col, session_gap, session_column_name):
    raise_dispatch_unregistered_type(X, kind="Dataframe")


@_add_session_column.specialize("pandas")
def _add_session_column_pandas(
    X, split_by, timestamp_col, session_gap, session_column_name
):
    # check if the time difference between events exceeds the session gap
    time_diff = X[timestamp_col].diff().dt.total_seconds().fillna(0) > session_gap
    if split_by:
        # check if the "split_by" column changes
        has_split_change = (X[split_by].diff().fillna(0) != 0).any(axis=1)
        # a new session starts if either the "split_by" column changes or the time
        # gap is exceeded
        is_new_session = has_split_change | time_diff
    else:
        is_new_session = time_diff
    # Compute cumulative sum of is_new_session to create session IDs
    return X.assign(**{session_column_name: is_new_session.cumsum()})


@_add_session_column.specialize("polars")
def _add_session_column_polars(
    X, split_by, timestamp_col, session_gap, session_column_name
):
    # check if the time difference between events exceeds the session gap
    time_diff = X[timestamp_col].diff().dt.total_seconds().fill_null(0) > session_gap
    if split_by:
        # check if the "split_by" column changes
        has_split_change = X.select(
            pl.any_horizontal(pl.col(split_by).diff().fill_null(0) != 0)
        ).to_series()
        # a new session starts if either the "split_by" column changes or the time
        # gap is exceeded
        is_new_session = has_split_change | time_diff
    else:
        is_new_session = time_diff
    # Add session_id by computing cumulative sum of is_new_session
    return X.with_columns(is_new_session.cum_sum().alias(session_column_name))


@dispatch
def _factorize_column(X, column_name):
    # Factorization is done so different groups can be found by doing a simple
    # numeric difference
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
    by at most ``session_gap`` seconds. Additionally, it is possible to provide a column
    or list of columns that can be used to distinguish between sessions, such
    as user identifiers (specified by the ``split_by`` column).
    When the time gap between consecutive events exceeds ``session_gap``, or
    when what identifies a user changes, a new session begins. All unrelated columns
    are passed through unchanged.

    Parameters
    ----------
    timestamp_col : str
        The name of the column that identifies the time of an event. This column
        is used to determine the start and end of a session.

    split_by : optional[str, list[str]], default=None
        The name of the column, or list of columns, to use to define sessions.
        A session boundary is created when the value in any of these columns
        changes, or when the time gap between events exceeds ``session_gap``.
        This is typically a user identifier column, but it can also be used to define
        sessions by other groupings (e.g. user and device type).
        If not provided, sessions are detected based on the time gap between events,
        and all events are considered to belong to the same user (or group).

    session_gap : int, default=1800
        The maximum gap (in seconds) between events in a session. If the gap
        between two events exceeds this value, they are considered to be in
        different sessions. Default is 1800 seconds (30 minutes).

    suffix : str, default="session_id"
        The suffix to be added to the name of the timestamp column.

    Attributes
    ----------
    all_inputs_ : list of str
        All column names in the input dataframe.

    all_outputs_: list of str
        All column names in the input dataframe plus the new column that identifies
        the session, with name "{timestamp}_{suffix}".

    Examples
    --------
    Consider this example where we have a dataframe with user events, and we want
    to identify sessions based on a 30-minute gap between events for each user.
    Users are identified by the value of the column ``user_id``.

    >>> import pandas as pd
    >>> from skrub import SessionEncoder
    >>> from datetime import datetime, timedelta
    >>> encoder = SessionEncoder(
    ...     split_by='user_id', timestamp_col='timestamp'
    ... )
    >>> data = {
    ...     'user_id': ['alice', 'alice', 'alice', 'bob', 'bob'],
    ...     'timestamp': [
    ...         pd.Timestamp('2024-01-01 10:00:00'),
    ...         pd.Timestamp('2024-01-01 10:05:00'),  # 5 min later, same session
    ...         pd.Timestamp('2024-01-01 11:00:00'),  # 55 min later, new session
    ...         pd.Timestamp('2024-01-01 10:00:00'),  # Different user
    ...         pd.Timestamp('2024-01-01 10:20:00'),  # 20 min later, same session
    ...     ],
    ...     'action': ['login', 'view', 'purchase', 'login', 'purchase']
    ... }
    >>> df = pd.DataFrame(data)
    >>> df
        user_id           timestamp   action
    0    alice 2024-01-01 10:00:00    login
    1    alice 2024-01-01 10:05:00     view
    2    alice 2024-01-01 11:00:00 purchase
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
    on different devices should have separate sessions.

    >>> encoder_multi = SessionEncoder(
    ...     split_by=['user_id', 'device_id'],
    ...     timestamp_col='timestamp',
    ... )
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
    user_id device_id           timestamp    action  timestamp_session_id
    0        1    mobile 2024-01-01 10:00:00      view                     1
    1        1    mobile 2024-01-01 10:10:00  purchase                     1
    2        1   desktop 2024-01-01 10:05:00      view                     0
    3        1   desktop 2024-01-01 10:20:00  checkout                     0
    4        2    mobile 2024-01-01 10:00:00     login                     2
    5        2    mobile 2024-01-01 10:15:00      view                     2

    In this example:

    - User 1 on "desktop" has session 0.
    - User 1 on "mobile" has session 1 (different device, so separate session).
    - User 2 on "mobile" has session 2 (different user).

    Note that sessions are defined by sorting over the ``split_by`` columns and then
    by the timestamp: this is why, while the "desktop"
    session of User 1 starts after their "mobile" session, it has session id ``0``
    since in alphabetical ordering "desktop" is first.

    You can also use ``SessionEncoder`` without a user identifier column. In this case,
    sessions are separated only by time gaps. This is useful for analyzing a single
    timeseries or events that don't have a user dimension:

    >>> encoder_no_split = SessionEncoder(
    ...     split_by=None,
    ...     timestamp_col='timestamp',
    ... )
    >>> data_no_split = {
    ...     'timestamp': [
    ...         pd.Timestamp('2024-01-01 10:00:00'),
    ...         pd.Timestamp('2024-01-01 10:10:00'),  # 10 min gap
    ...         pd.Timestamp('2024-01-01 10:15:00'),  # 5 min gap, still in session
    ...         pd.Timestamp('2024-01-01 11:00:00'),  # 45 min gap, new session
    ...         pd.Timestamp('2024-01-01 11:10:00'),  # 10 min gap, continue session
    ...     ],
    ...     'event_type': ['start', 'action', 'action', 'restart', 'action']
    ... }
    >>> df_no_split = pd.DataFrame(data_no_split)
    >>> result_no_split = encoder_no_split.fit_transform(df_no_split)
    >>> result_no_split
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

    It is possible to change the duration of the session gap by setting the
    ``session_gap`` parameter. For example, we can set it to 5 minutes (300 seconds)
    instead of the default 30 minutes, and this will change the session assignments
    accordingly:

    >>> encoder_new_gap = SessionEncoder(
    ...     split_by=None,
    ...     timestamp_col='timestamp',
    ...     session_gap=300
    ... )
    >>> result_new_gap = encoder_new_gap.fit_transform(df_no_split)
    >>> result_new_gap
                timestamp event_type  timestamp_session_id
    0 2024-01-01 10:00:00      start                     0
    1 2024-01-01 10:10:00     action                     1
    2 2024-01-01 10:15:00     action                     1
    3 2024-01-01 11:00:00    restart                     2
    4 2024-01-01 11:10:00     action                     3

    It is also possible to change the suffix that is added at the end of the session
    ID column via the "suffix" parameter. This is useful, for example, if you want
    to add sessions based on different groupings or intervals:

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
    >>> df = pd.DataFrame(data_multi)
    >>> encoder_user = SessionEncoder("timestamp",
    ... split_by=["user_id"], suffix="user")
    >>> encoder_user.fit_transform(df)
    user_id device_id           timestamp    action  timestamp_user
    0        1    mobile 2024-01-01 10:00:00      view                  0
    1        1    mobile 2024-01-01 10:10:00  purchase                  0
    2        1   desktop 2024-01-01 10:05:00      view                  0
    3        1   desktop 2024-01-01 10:20:00  checkout                  0
    4        2    mobile 2024-01-01 10:00:00     login                  1
    5        2    mobile 2024-01-01 10:15:00      view                  1

    >>> encoder_user_device = SessionEncoder("timestamp",
    ... split_by=["user_id", "device_id"],
    ... suffix="user_device")
    >>> encoder_user_device.fit_transform(df)
    user_id device_id           timestamp    action  timestamp_user_device
    0        1    mobile 2024-01-01 10:00:00      view                      1
    1        1    mobile 2024-01-01 10:10:00  purchase                      1
    2        1   desktop 2024-01-01 10:05:00      view                      0
    3        1   desktop 2024-01-01 10:20:00  checkout                      0
    4        2    mobile 2024-01-01 10:00:00     login                      2
    5        2    mobile 2024-01-01 10:15:00      view                      2

    """

    def __init__(
        self, timestamp_col, split_by=None, session_gap=30 * 60, suffix="session_id"
    ):
        self.timestamp_col = timestamp_col
        self.split_by = split_by
        self.session_gap = session_gap
        self.suffix = suffix

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

        # Checking that all the needed columns are there
        self._check_input_dataframe()
        # check the correctness of the values of session_gap
        if not isinstance(self.session_gap, numbers.Number):
            raise TypeError(f"Expected a number, got {type(self.session_gap)}")
        if self.session_gap <= 0:
            raise ValueError(
                f"session_gap must be a positive number, got {self.session_gap}"
            )
        if not isinstance(self.suffix, str) or self.suffix is None:
            raise ValueError(f"Expected a string as suffix, got {self.suffix!r}")

        self._session_id_name = f"{self.timestamp_col}_{self.suffix}"

        # If the generated session id column name already exists in the input dataframe,
        # we add a random suffix to avoid overwriting it
        if self._session_id_name in self.all_inputs_:
            self._session_id_name += f"_skrub_{random_string()}"

        # if the input dataframe is empty, we can skip all the processing and
        # return an empty dataframe with the session_id column added
        if sbd.is_empty_frame(X):
            X = sbd.with_columns(
                X, **{self._session_id_name: np.array([], dtype=np.float32)}
            )
            return X

        # Adding a row order column to sort lines back
        row_order_col = f"_row_order_skrub_{random_string()}"
        X = sbd.with_columns(X, **{row_order_col: range(X.shape[0])})

        # Dropping unneeded columns to reduce the sorting overhead
        if cols_to_remove := [
            _
            for _ in self.all_inputs_
            if _ not in self._split_by_columns + [self.timestamp_col]
        ]:
            X_selected = sbd.drop_columns(X, cols_to_remove)
        else:
            X_selected = X

        # sort the input dataframe by the "split_by" and "timestamp" columns
        sort_by = (
            self._split_by_columns + [self.timestamp_col]
            if self.split_by is not None
            else [self.timestamp_col]
        )
        X_sorted = sbd.sort(X_selected, by=sort_by)

        X_factorized, factorized_by = self._factorize_columns(X_sorted)

        X_with_session_id = self._add_session_id(
            X_factorized,
            factorized_by,
        )
        # Reordering rows back to the original order
        X_result = sbd.sort(X_with_session_id, by=row_order_col)

        # drop the factorized "split_by" columns if the original "split_by"
        # columns were not numeric, and the column used to reorder
        to_drop = [col for col in factorized_by if col not in self._split_by_columns]
        to_drop += [row_order_col]
        X_result = sbd.drop_columns(X_result, to_drop)

        # If unrelated columns were removed earlier, bring them back here
        if cols_to_remove:
            X_result = sbd.concat(X_result, s.select(X, cols_to_remove), axis=1)

        # Reordering columns so that the session_id is added as the last column
        X_result = s.select(X_result, self.all_inputs_ + [self._session_id_name])

        self.all_outputs_ = sbd.column_names(X_result)
        return X_result

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

    def _check_input_dataframe(self):
        """
        Check that the input columns are present and correct
        """

        # Check that the timestamp column is present
        if self.timestamp_col not in self.all_inputs_:
            raise ValueError(
                f"Column '{self.timestamp_col}' not found in input dataframe"
            )
        # check that the required columns are present in the input dataframe
        self._split_by_columns = []
        if self.split_by is not None:
            if isinstance(self.split_by, str):
                self._split_by_columns = [self.split_by]
            elif isinstance(self.split_by, Iterable) and not isinstance(
                self.split_by, str
            ):
                self._split_by_columns = list(self.split_by)
            else:
                raise TypeError("split_by must be a string, a list of strings, or None")
            for col in self._split_by_columns:
                if col not in self.all_inputs_:
                    raise ValueError(f"Column '{col}' not found in input dataframe")

    def _factorize_columns(self, X):
        """
        convert split_by columns to numerical columns if they're not already, to
        ensure that the diff operation works correctly
        """

        if not self.split_by:
            return X, []
        factorized_columns = {
            f"{col}_factorized_skrub_{random_string()}": _factorize_column(X, col)
            if not sbd.is_numeric(X[col])
            else X[col]
            for col in self._split_by_columns
        }

        X_factorized = sbd.with_columns(X, **factorized_columns)

        return X_factorized, list(factorized_columns.keys())

    def _add_session_id(self, X_factorized, factorized_by):
        X_with_session_id = _add_session_column(
            X_factorized,
            factorized_by,
            self.timestamp_col,
            self.session_gap,
            self._session_id_name,
        )
        return X_with_session_id

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
