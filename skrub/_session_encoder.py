"""
The SessionEncoder is a transformer that takes as input:
- a "by" column, which identifies a user
- a "timestamp" column, which identifies the time of an event
- a "session_duration" value, which identifies the duration of a session

It returns a dataframe with the same number of rows as the input, but with the following
columns:
- "session_id": a unique identifier for each session, which is a combination of the "by"
column and a session number
- "session_start": the timestamp of the first event in the session
- "session_end": the timestamp of the last event in the session
- "session_duration": the duration of the session, which is the difference between the
last and first timestamps in the session
"""

from sklearn.base import BaseEstimator, TransformerMixin

from . import _dataframe as sbd


class SessionEncoder(TransformerMixin, BaseEstimator):
    """Encode sessions from a dataframe.

    Parameters
    ----------
    by : str
        The name of the column that identifies a user. This column is used to
        group events into sessions.

    timestamp : str
        The name of the column that identifies the time of an event. This column
        is used to determine the start and end of a session.

    session_duration : str, optional
        The name of the column that identifies the duration of a session. If not
        provided, the duration is calculated as the difference between the last
        and first timestamps in the session.

    session_gap : int, default=30
        The maximum gap (in minutes) between events in a session. If the gap
        between two events exceeds this value, they are considered to be in
        different sessions.

    Attributes
    ----------
    all_inputs_ : list of str
        All column names in the input dataframe.
    """

    def __init__(self, by, timestamp, session_duration=None, session_gap=30):
        self.by = by
        self.timestamp = timestamp
        self.session_duration = session_duration
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

        # check the correctness of the values of session_gap and session_duration
        if not isinstance(self.session_gap, (int, float)) or self.session_gap <= 0:
            raise ValueError("session_gap must be a positive number")

        if self.session_duration is not None and not isinstance(
            self.session_duration, (int, float)
        ):
            raise ValueError("session_duration must be a number if provided")

        # sort the input dataframe by the "by" and "timestamp" columns
        X_sorted = sbd.sort(X, by=[self.by, self.timestamp])  # noqa
        # mark the start of a new session by checking the difference

        # add the session id

        # compute statistics

        # wrap everything up in a dataframe and return it

        return self.transform(X)

    def _check_is_new_session(self, X):
        """Check if a new session starts at each row of the dataframe.

        Parameters
        ----------
        X : pandas.DataFrame or polars.DataFrame
            The input dataframe.

        Returns
        -------
        pandas.Series or polars.Series
            A boolean series indicating whether a new session starts at each row.
        """
        # check if the "by" column changes
        char_diff = X[self.by].diff().fillna(0) > 0
        # check if the time difference between events exceeds the session gap
        time_diff = X[self.timestamp].diff().fillna(0) > self.session_gap * 60 * 1000
        # a new session starts if either the "by" column changes or the time gap is
        # exceeded
        is_new_session = char_diff | time_diff
        return is_new_session
