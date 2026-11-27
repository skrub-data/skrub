import numpy as np
import pandas as pd

from . import _dataframe as sbd
from ._dispatch import dispatch, raise_dispatch_unregistered_type


@dispatch
def _split_boundaries(time, train, test):
    raise_dispatch_unregistered_type(time, train, test, kind="Series")


@_split_boundaries.specialize("pandas", argument_type="Column")
def _split_boundaries_pandas(time, train, test):
    train_mask = time.between(train[0], train[1], inclusive="left").to_numpy()
    train_indices = np.where(train_mask)[0]

    test_mask = time.between(test[0], test[1], inclusive="left").to_numpy()
    test_indices = np.where(test_mask)[0]

    return train_indices, test_indices


@_split_boundaries.specialize("polars", argument_type="Column")
def _split_boundaries_polars(time, train, test):
    train_mask = time.is_between(train[0], train[1], closed="left").to_numpy()
    train_indices = np.where(train_mask)[0]

    test_mask = time.is_between(test[0], test[1], closed="left").to_numpy()
    test_indices = np.where(test_mask)[0]

    return train_indices, test_indices


class TimeSeriesSplitter:
    """
    Time series cross-validation splitter with configurable gaps.

    Generates train/test splits for time series data with optional gaps
    between the train and test periods. Splits are right-aligned (newest
    first).

    The possible period sizes for ``min_train_size``, ``test_size`` and
    ``gap_size`` use the pandas offset aliases syntax; see the
    `pandas documentation on offset aliases
    <https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases>`_.

    Parameters
    ----------
    time_col : str
        Name of the datetime column to use for splitting.
    min_train_size : str
        Minimum size of the train period, as a pandas frequency string
        (e.g., '90D', '3ME', '1Y').
    test_size : str
        Duration of each test period, as a pandas frequency string.
    gap_size : str, default='0D'
        Gap between train and test periods, as a pandas frequency string.
        Use '0D' for no gap.
    max_nsplits : int, default=5
        Maximum number of splits to generate. Takes the newest splits.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    >>> df = pd.DataFrame({'date': dates, 'value': np.random.randn(len(dates))})
    >>> splitter = TimeSeriesSplitter(
    ...     time_col='date',
    ...     min_train_size='90D',
    ...     test_size='30D',
    ...     gap_size='7D',
    ...     max_nsplits=5
    ... )
    >>> for train_idx, test_idx in splitter.split(df):
    ...     train_data = df.iloc[train_idx]
    ...     test_data = df.iloc[test_idx]
    """

    def __init__(
        self,
        time_col,
        min_train_size,
        test_size,
        gap_size="0D",
        max_nsplits=5,
    ):
        self.time_col = time_col
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.gap_size = gap_size
        self.gap = pd.tseries.frequencies.to_offset(self.gap_size)
        self.max_nsplits = max_nsplits

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into train and test sets.

        Parameters
        ----------
        X : DataFrame
            Data to split (pandas or polars DataFrame).
        y : array-like, optional
            Target variable (not used, for sklearn compatibility).
        groups : array-like, optional
            Group labels (not used, for sklearn compatibility).

        Yields
        ------
        train_indices : ndarray
            Integer indices for the train set.
        test_indices : ndarray
            Integer indices for the test set.
        """
        time = sbd.col(X, self.time_col)
        min_date = sbd.min(time)
        max_date = sbd.max(time)

        boundaries = self.get_boundaries(min_date, max_date)

        for train, gap_period, test in boundaries[: self.max_nsplits]:
            train_indices, test_indices = _split_boundaries(time, train, test)

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Return the number of splitting iterations.

        Parameters
        ----------
        X : DataFrame
            Data to split. Required to determine date range.
        y : array-like, optional
            Not used, for sklearn compatibility.
        groups : array-like, optional
            Not used, for sklearn compatibility.

        Returns
        -------
        n_splits : int
            Number of train/test splits.
        """
        if X is None:
            raise ValueError("X is required to determine n_splits")

        return len(self.split)

    def get_boundaries(self, min_date, max_date):
        """
        Compute train/gap/test boundaries for all splits.

        Parameters
        ----------
        min_date : datetime-like
            Earliest date in the dataset.
        max_date : datetime-like
            Latest date in the dataset.

        Returns
        -------
        boundaries : list of tuple
            List of (train, gap_period, test) tuples, where each element
            is a tuple of (start_date, end_date).
        """
        min_date = pd.Timestamp(min_date)
        max_date = pd.Timestamp(max_date)
        min_train = min_date + pd.tseries.frequencies.to_offset(self.min_train_size)
        dateranges = pd.date_range(
            start=max_date,
            end=min_train + self.gap,
            freq=f"-{self.test_size}",
        )

        # make a split that includes the newest data even if the
        # input frequency does not naturally land on the newest
        # date - although size of this last split < test_size
        if len(dateranges) == 0 or dateranges[0] != max_date:
            dateranges = pd.DatetimeIndex([max_date]).union(dateranges, sort=False)

        dateranges = dateranges.to_pydatetime()
        min_date = min_date.to_pydatetime()

        boundaries = []
        for start, end in zip(dateranges[1:], dateranges[:-1]):
            train = (min_date, start - self.gap)
            gap_period = (start - self.gap, start)
            test = (start, end)

            boundaries.append((train, gap_period, test))

        return boundaries
