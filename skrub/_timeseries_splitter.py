import numpy as np
import pandas as pd

# from ._dispatch import dispatch, raise_dispatch_unregistered_type


class TimeSeriesSplitter:
    def __init__(
        self,
        time_col="date",
        min_training_size="3ME",
        test_frequency="1ME",
        gap_size=0,
        max_nsplits=5,
        split_sampling="newest",
    ):
        self.time_col = time_col
        self.min_training_size = min_training_size
        self.test_frequency = test_frequency
        self.gap_size = pd.Timedelta(days=gap_size)
        self.max_nsplits = max_nsplits
        self.split_sampling = split_sampling

    def split(self, X, y=None, groups=None):
        min_date = X[self.time_col].min()
        max_date = X[self.time_col].max()
        min_train = min_date + pd.tseries.frequencies.to_offset(self.min_training_size)

        split_boundaries = self.get_boundaries(min_date, max_date, min_train)

        for train, gap_period, test in split_boundaries:
            # Use inclusive='left' to avoid boundary overlaps ([start, end))
            train_mask = X[self.time_col].between(train[0], train[1], inclusive="left")
            train_indices = np.where(train_mask)[0]

            # gap_mask = X[self.time_col].between(gap_period[0], gap_period[1],
            #                                               inclusive='left')
            # gap_indices = np.where(gap_mask)[0]

            test_mask = X[self.time_col].between(test[0], test[1], inclusive="left")
            test_indices = np.where(test_mask)[0]

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        if X is None:
            raise ValueError("X is required to determine n_splits")

        min_date = X[self.time_col].min()
        max_date = X[self.time_col].max()
        min_train = min_date + pd.tseries.frequencies.to_offset(self.min_training_size)

        return len(self.get_boundaries(min_date, max_date, min_train))

    def get_boundaries(self, min_date, max_date, min_train):
        test_dates = pd.date_range(
            start=max_date,
            end=min_train + self.gap_size,
            freq=f"-{self.test_frequency}",
        )

        if len(test_dates) == 0 or test_dates[0] != max_date:
            test_dates = pd.DatetimeIndex([max_date]).union(test_dates, sort=False)

        boundaries = []

        for start, end in zip(test_dates[1:], test_dates[:-1]):
            train = (min_date, start - self.gap_size)
            gap_period = (start - self.gap_size, start)
            test = (start, end)

            boundaries.append((train, gap_period, test))

        return boundaries
