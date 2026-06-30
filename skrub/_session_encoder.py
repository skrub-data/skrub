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

try:
    import polars as pl
except ImportError:
    pl = None

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from . import selectors as s
from ._dispatch import dispatch, raise_dispatch_unregistered_type
from ._utils import random_string


@dispatch
def _add_session_column(
    X, split_by_columns, timestamp_column, session_gap, session_id_column
):
    raise_dispatch_unregistered_type(X, kind="Dataframe")


@_add_session_column.specialize("pandas")
def _add_session_column_pandas(
    X, split_by_columns, timestamp_column, session_gap, session_id_column
):
    # Adding a row order column to sort lines back
    row_order_col = f"_row_order_skrub_{random_string()}"
    X_with_order = X.assign(**{row_order_col: range(X.shape[0])})

    # Selecting only the columns needed for sessionization and sorting them
    # to ensure that the sessionization is done correctly

    selected = split_by_columns + [timestamp_column]
    X_selected = s.select(X_with_order, selected + [row_order_col])
    X_has_nulls = X_selected.loc[X_selected[selected].isnull().any(axis=1)]
    # Assigning a session ID of -1 to rows with nulls in timestamp or group_by columns
    # -1 rather than None because adding nulls to a pandas column of integers will
    # convert it to float
    # This is a problem because session IDs are meant to be grouped over, and
    # when the session ID is float32, numerical instability can cause issues
    # with grouping if there are a lot of sessions
    X_has_nulls = X_has_nulls.assign(**{session_id_column: -1})

    X_selected = X_selected.dropna(subset=selected)

    # needed to avoid a warning with min deps
    grouper = split_by_columns[0] if len(split_by_columns) == 1 else split_by_columns
    groups = (
        X_selected.groupby(grouper) if len(split_by_columns) > 0 else [("", X_selected)]
    )
    rolling_session_id = 0

    groups_with_session_ids = []

    for _, group_df in groups:
        group_df_sorted = group_df.sort_values(by=timestamp_column)
        # Compute time differences between consecutive events
        time_diffs = group_df_sorted[timestamp_column].diff().dt.total_seconds()
        # Identify session boundaries based on time gaps
        session_boundaries = (time_diffs > session_gap) | (time_diffs.isna())
        # Assign session IDs based on cumulative sum of session boundaries
        # cumsum - 1 to start session IDs at 0
        session_ids = session_boundaries.cumsum() - 1 + rolling_session_id
        # Update rolling_session_id for the next group
        rolling_session_id = session_ids.max() + 1

        group_df_sorted = group_df_sorted.assign(
            **{
                session_id_column: pd.Series(
                    session_ids.values, index=group_df_sorted.index
                )
            }
        )
        groups_with_session_ids.append(group_df_sorted)
    X_with_session_id = pd.concat(groups_with_session_ids, axis=0)

    X_with_session_id = pd.concat([X_with_session_id, X_has_nulls], axis=0)

    # Reordering rows back to the original order and selecting session id
    return X_with_session_id.sort_values(by=row_order_col)[session_id_column]


@_add_session_column.specialize("polars")
def _add_session_column_polars(
    X, split_by_columns, timestamp_column, session_gap, session_id_column
):
    # Adding a row order column to sort lines back
    row_order_col = f"_row_order_skrub_{random_string()}"
    X_with_order = X.with_row_index(name=row_order_col)

    # Selecting only the columns needed for sessionization and sorting them
    # to ensure that the sessionization is done correctly
    X_selected = s.select(
        X_with_order, split_by_columns + [timestamp_column, row_order_col]
    )

    # Identify rows with nulls in timestamp or group_by columns
    selected = split_by_columns + [timestamp_column]

    # Find rows with nulls in timestamp or group_by columns and assign them a session
    # ID of -1
    # -1 rather than None for consistency with pandas implementation
    X_has_nulls = X_selected.filter(
        pl.any_horizontal(pl.col(selected).is_null())
    ).with_columns(pl.lit(-1).cast(pl.Int64).alias(session_id_column))

    X_selected = X_selected.drop_nulls(subset=selected)

    groups = (
        X_selected.group_by(split_by_columns, maintain_order=True)
        if len(split_by_columns) > 0
        else [("", X_selected)]
    )
    rolling_session_id = 0

    groups_with_session_ids = []

    for _, group_df in groups:
        group_df_sorted = group_df.sort(by=timestamp_column)
        # Compute time differences between consecutive events
        time_diffs = group_df_sorted[timestamp_column].diff().dt.total_seconds()
        # Identify session boundaries based on time gaps
        session_boundaries = (time_diffs > session_gap) | (
            # need both is_nan and is_null to handle older versions of polars
            time_diffs.is_nan() | time_diffs.is_null()
        ).fill_null(True)
        # Assign session IDs based on cumulative sum of session boundaries
        # cumsum - 1 to start session IDs at 0
        session_ids = session_boundaries.cum_sum() - 1 + rolling_session_id
        # Update rolling_session_id for the next group
        rolling_session_id = session_ids.max() + 1

        group_df_sorted = group_df_sorted.with_columns(
            session_ids.alias(session_id_column)
        )
        groups_with_session_ids.append(group_df_sorted)
    X_with_session_id = pl.concat(groups_with_session_ids)
    X_with_session_id = X_with_session_id.with_columns(
        pl.col(session_id_column).cast(pl.Int64)
    )

    # Concatenate rows with nulls back
    X_with_session_id = pl.concat([X_with_session_id, X_has_nulls])

    # Reordering rows back to the original order and selecting only the session id
    return X_with_session_id.sort(by=row_order_col)[session_id_column]


class SessionEncoder(TransformerMixin, BaseEstimator):
    """Add a session ID column to a dataframe based on time gaps and other columns.

    A session is defined as a sequence of events  where consecutive events are separated
    by at most ``session_gap`` seconds. Additionally, it is possible to provide a column
    or list of columns that can be used to distinguish between sessions, such
    as user identifiers (specified by the ``split_by`` column).
    Within each sequence identified by a unique value in the ``split_by`` column(s),
    a new session is started when the time gap between events exceeds ``session_gap``
    seconds.

    The encoder takes care of grouping the data by ``split_by`` and sorting by
    timestamp column before identifying sessions, and sorting it back to the
    original order at the end, so the original order of events in the input
    dataframe does not matter.

    If a null value is present in the timestamp column or any of the ``split_by``
    columns, the corresponding row will be assigned a session ID of -1, and will
    be ignored when computing time intervals.

    All unrelated columns are passed through unchanged.

    Parameters
    ----------
    timestamp_col : str
        The name of the column that identifies the time of an event. This column
        is used to determine the start and end of a session. ``timestamp_col`` must
        be a datetime, and an error will be raised otherwise.
        Sessions are defined within each group of events that have the same value
        in the ``split_by`` column(s) (or all events if ``split_by`` is None), and
        the result will be correct no matter the order of the rows in the input
        dataframe.

    split_by : optional[str, list[str]], default=None
        The name of the column (typically the user ID), or list of columns, that
        is used to identify independent event sequence (such as the activity of
        different users).
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
        The suffix to be added to the name of the created session id column.

    Attributes
    ----------
    all_inputs_ : list of str
        All column names in the input dataframe.

    all_outputs_: list of str
        All column names in the input dataframe plus the new column that identifies
        the session, with name "{timestamp}_{suffix}".

    session_id_column_ : str
        The name of the session ID column that is added to the dataframe. This is
        generated as "{timestamp_col}_{suffix}", but if this name already exists in
        the input dataframe, a random suffix is added to avoid overwriting it.

    Examples
    --------
    Consider this example where we have a dataframe with user events, and we want
    to identify sessions based on a 30-minute gap between events for each user.
    Users are identified by the value of the column ``user_id``.
    Note that the order of the rows in the input dataframe does not matter.

    Sessions are defined by sorting over the ``split_by``columns (if provided)
    and then by the timestamp.

    >>> import pandas as pd
    >>> from datetime import datetime, timedelta
    >>> data = {
    ...     "user_id": [1, 1, 1, 1, 1, 2, 2],
    ...     "device_id": [
    ...         "mobile",
    ...         "mobile",
    ...         "desktop",
    ...         "desktop",
    ...         "mobile",
    ...         "mobile",
    ...         "mobile",
    ...     ],
    ...     "timestamp": [
    ...         pd.Timestamp("2024-01-01 10:00:00"),
    ...         pd.Timestamp("2024-01-01 10:10:00"),  # 10 min later, same session
    ...         pd.Timestamp("2024-01-01 10:05:00"),  # Different device (sorted),
    ...                                                 # different session
    ...     pd.Timestamp("2024-01-01 10:20:00"),  # 15 min later, same session
    ...                                                 # different session
    ...     pd.Timestamp("2024-01-01 11:20:00"),  # 60 min later, new session
    ...     pd.Timestamp("2024-01-01 10:00:00"),  # Different user
    ...     pd.Timestamp("2024-01-01 10:15:00"),  # 15 min later, same session
    ... ],
    ...     "action": [
    ...         "view",
    ...         "purchase",
    ...         "view",
    ...         "add_to_cart",
    ...         "checkout",
    ...         "view",
    ...         "wishlist",
    ...     ],
    ... }
    >>> df = pd.DataFrame(data)
    >>> df
    user_id device_id           timestamp       action
    0        1    mobile 2024-01-01 10:00:00         view
    1        1    mobile 2024-01-01 10:10:00     purchase
    2        1   desktop 2024-01-01 10:05:00         view
    3        1   desktop 2024-01-01 10:20:00  add_to_cart
    4        1    mobile 2024-01-01 11:20:00     checkout
    5        2    mobile 2024-01-01 10:00:00         view
    6        2    mobile 2024-01-01 10:15:00     wishlist

    We use the ``SessionEncoder`` with default ``session_gap`` of 30 minutes:

    >>> from skrub import SessionEncoder
    >>> encoder = SessionEncoder(
    ...     split_by='user_id', timestamp_col='timestamp'
    ... )
    >>> result = encoder.fit_transform(df)
    >>> result
    user_id device_id           timestamp       action  timestamp_session_id
    0        1    mobile 2024-01-01 10:00:00         view                     0
    1        1    mobile 2024-01-01 10:10:00     purchase                     0
    2        1   desktop 2024-01-01 10:05:00         view                     0
    3        1   desktop 2024-01-01 10:20:00  add_to_cart                     0
    4        1    mobile 2024-01-01 11:20:00     checkout                     1
    5        2    mobile 2024-01-01 10:00:00         view                     2
    6        2    mobile 2024-01-01 10:15:00     wishlist                     2

    In this example, grouping by `user_id` results in three separate sessions:

    - User 1 has two sessions (session 0 and session 1) because there is a gap of
      60 minutes between their events at 10:20 and 11:20, which exceeds the 30-minute
      threshold. The first four events of user 1 belong to session 0, while the
      last event belongs to session 1.
    - User 2 has one session (session 2) because all their events are within
      30 minutes of the previous one.

    You can also identify users by multiple columns. For instance, the same user
    on different devices should have separate sessions.

    >>> encoder_multi = SessionEncoder(
    ...     split_by=['user_id', 'device_id'],
    ...     timestamp_col='timestamp',
    ... )
    >>> result_multi = encoder_multi.fit_transform(df)
    >>> result_multi
    user_id device_id           timestamp       action  timestamp_session_id
    0        1    mobile 2024-01-01 10:00:00         view                     1
    1        1    mobile 2024-01-01 10:10:00     purchase                     1
    2        1   desktop 2024-01-01 10:05:00         view                     0
    3        1   desktop 2024-01-01 10:20:00  add_to_cart                     0
    4        1    mobile 2024-01-01 11:20:00     checkout                     2
    5        2    mobile 2024-01-01 10:00:00         view                     3
    6        2    mobile 2024-01-01 10:15:00     wishlist                     3

    In this example:

    - User 1 on "desktop" has session 0.
    - User 1 on "mobile" has two sessions, session 1 and session 2, because there
      is a gap of 60 minutes between their events at 10:10 and 11:20, which exceeds
      the 30-minute threshold.
    - User 2 on "mobile" has session 3 (different user).

    Note that the value of the session IDs are arbitrary and depend on the order
    of events in the dataframe. The important thing is that events that belong to
    the same session have the same session ID, and events that belong to different
    sessions have different session IDs.

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

    When the timestamp column or any of the split_by columns contains null values,
    those rows will be assigned **session ID -1**, and will be ignored when computing
    time intervals. In some versions of pandas,
    adding nulls to a pandas column of integers will convert it to float, which can
    cause issues with grouping if there are a lot of sessions.

    >>> data_with_nulls = {
    ...     'user_id': [1, 1, None, 1],  # None value in split_by column
    ...     'timestamp': [
    ...         pd.Timestamp('2024-01-01 10:00:00'),
    ...         None,  # None value in timestamp column
    ...         pd.Timestamp('2024-01-01 10:10:00'),
    ...         pd.Timestamp('2024-01-01 10:20:00'),
    ...     ],
    ... }
    >>> df_with_nulls = pd.DataFrame(data_with_nulls)
    >>> encoder_with_nulls = SessionEncoder(
    ...     split_by='user_id',
    ...     timestamp_col='timestamp'
    ... )
    >>> result_with_nulls = encoder_with_nulls.fit_transform(df_with_nulls)
    >>> result_with_nulls
    user_id           timestamp  timestamp_session_id
    0      1.0 2024-01-01 10:00:00                     0
    1      1.0                 NaT                    -1
    2      NaN 2024-01-01 10:10:00                    -1
    3      1.0 2024-01-01 10:20:00                     0

    In this example:

    - Row 0 has valid user_id and timestamp, so it gets session ID 0.
    - Row 1 has a null timestamp, so it gets session ID -1.
    - Row 2 has a null user_id, so it gets session ID -1.
    - Row 3 has valid user_id and timestamp, and since there is less than 30 minutes
      between rows 0 and 3 (ignoring row 1 and 2 due to null values), it gets the
      same session ID as row 0.




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
        self._check_input(X)

        self.session_id_column_ = f"{self.timestamp_col}_{self.suffix}"

        # If the generated session id column name already exists in the input dataframe,
        # we add a random suffix to avoid overwriting it
        if self.session_id_column_ in self.all_inputs_:
            self.session_id_column_ += f"_skrub_{random_string()}"

        return self.transform(X, y)

    def transform(self, X, y=None):
        """Transform the data by encoding sessions.

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
        check_is_fitted(self)

        # if the input dataframe is empty, we can skip all the processing and
        # return an empty dataframe with the session_id column added
        if sbd.is_empty_frame(X):
            return sbd.with_columns(
                X, **{self.session_id_column_: np.array([], dtype=np.int64)}
            )

        session_id = _add_session_column(
            X,
            self._split_by_columns,
            self.timestamp_col,
            self.session_gap,
            self.session_id_column_,
        )
        X_result = sbd.with_columns(X, **{self.session_id_column_: session_id})

        self.all_outputs_ = sbd.column_names(X_result)
        return X_result

    def _check_input(self, X):
        """
        Check that the input columns are present and correct
        """
        # check the correctness of the values of session_gap
        if not isinstance(self.session_gap, numbers.Number):
            raise TypeError(f"Expected a number, got {type(self.session_gap)}")
        if self.session_gap <= 0:
            raise ValueError(
                f"session_gap must be a positive number, got {self.session_gap}"
            )
        # check that the suffix is a string
        if not isinstance(self.suffix, str):
            raise TypeError(f"Expected a string as suffix, got {self.suffix!r}")

        # Check that the timestamp column is present
        if self.timestamp_col not in self.all_inputs_:
            raise ValueError(
                f"Column {self.timestamp_col!r} not found in input dataframe"
            )
        # check that the timestamp column is of datetime type
        if not sbd.is_empty_frame(X) and not sbd.is_any_date(
            sbd.col(X, self.timestamp_col)
        ):
            raise TypeError(
                "Expected a datetime column for timestamp_col,"
                f" got {sbd.dtype(sbd.col(X, self.timestamp_col))}"
            )

        # check that the required columns are present in the input dataframe
        if self.split_by is None:
            self._split_by_columns = []
            return
        if isinstance(self.split_by, str):
            self._split_by_columns = [self.split_by]
        elif isinstance(self.split_by, Iterable):
            self._split_by_columns = list(self.split_by)
        else:
            raise TypeError(
                "split_by must be a string, a list of strings, or None, got"
                f" {type(self.split_by)}"
            )
        for col in self._split_by_columns:
            if col not in self.all_inputs_:
                raise ValueError(f"Column {col!r} not found in input dataframe")

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
