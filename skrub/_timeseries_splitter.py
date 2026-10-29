import numpy as np
import pandas as pd

from . import _dataframe as sbd
from ._dispatch import dispatch, raise_dispatch_unregistered_type


@dispatch
def _split_boundaries(col):
    raise_dispatch_unregistered_type(col, kind="Series")


@_split_boundaries.specialize("pandas", argument_type="Column")
def _split_boundaries_pandas(time, train, test):
    train_mask = time.between(train[0], train[1], inclusive="left")
    train_indices = np.where(train_mask)[0]

    test_mask = time.between(test[0], test[1], inclusive="left")
    test_indices = np.where(test_mask)[0]

    return train_indices, test_indices


@_split_boundaries.specialize("polars", argument_type="Column")
def _split_boundaries_polars(time, train, test):
    train_mask = time.is_between(train[0], train[1], closed="left")
    train_indices = np.where(train_mask)[0]

    test_mask = time.is_between(test[0], test[1], closed="left")
    test_indices = np.where(test_mask)[0]

    return train_indices, test_indices


class TimeSeriesSplitter:
    """
    Time series cross-validation splitter with configurable gaps.

    Generates train/test splits for time series data with optional gaps
    between training and test periods. Splits are right-aligned (newest first).

    Parameters
    ----------
    time_col : str, default='date'
        Name of the datetime column to use for splitting.
    min_training_size : str, default='3ME'
        Minimum size of the training period, as a pandas frequency string
        (e.g., '90D', '3ME', '1Y').
    test_frequency : str, default='1ME'
        Duration of each test period, as a pandas frequency string.
    gap_size : str, default='0D'
        Gap between train and test periods, as a pandas frequency string.
        Use '0D' for no gap.
    max_nsplits : int, default=5
        Maximum number of splits to generate. Takes the newest splits.

    Examples
    --------
    >>> splitter = TimeSeriesSplitter(
    ...     time_col='date',
    ...     min_training_size='90D',
    ...     test_frequency='30D',
    ...     gap_size='7D',
    ...     max_nsplits=5
    ... )
    >>> for train_idx, test_idx in splitter.split(df):
    ...     train_data = df.iloc[train_idx]
    ...     test_data = df.iloc[test_idx]
    """

    def __init__(
        self,
        time_col="date",
        min_training_size="3ME",
        test_frequency="1ME",
        gap_size="0D",
        max_nsplits=5,
    ):
        self.time_col = time_col
        self.min_training_size = min_training_size
        self.test_frequency = test_frequency
        self.gap_size = gap_size
        self.gap = pd.tseries.frequencies.to_offset(self.gap_size)
        self.max_nsplits = max_nsplits

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test sets.

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
            Integer indices for the training set.
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

        time = sbd.col(X, self.time_col)
        min_date = sbd.min(time)
        max_date = sbd.max(time)

        return len(self.get_boundaries(min_date, max_date)[: self.max_nsplits])

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
        min_train = min_date + pd.tseries.frequencies.to_offset(self.min_training_size)
        dateranges = pd.date_range(
            start=max_date,
            end=min_train + self.gap,
            freq=f"-{self.test_frequency}",
        )

        # make a split that includes the newest data even if the
        # input frequency does not naturally land on the newest
        # date - although size of this last split < test_frequency
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

    def plot_splits(self, X, ax=None, **kwargs):
        """
        Visualize train/gap/test splits as horizontal bars.

        Parameters
        ----------
        X : DataFrame
            Data containing the time column.
        ax : matplotlib axes, optional
            Axes to plot on. If None, creates new figure.
        **kwargs : dict
            Styling parameters passed to format dictionary:
            - train_color : str, default='steelblue'
            - gap_color : str, default='lightcoral'
            - test_color : str, default='forestgreen'
            - alpha : float, default=0.5
            - gap_hatch : str, default='////'
            - gap_facecolor : str, default='white'
            - loc : str, default='upper right' (legend location)

        Returns
        -------
        fig : matplotlib figure
            The figure containing the plot.
        ax : matplotlib axes
            The axes with the plot.
        """
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))
        else:
            fig = ax.figure

        format = {
            "train_color": "steelblue",
            "gap_color": "lightcoral",
            "test_color": "forestgreen",
            "alpha": 0.5,
            "gap_hatch": "////",
            "gap_facecolor": "white",
            "loc": "upper right",
        }
        format.update(kwargs)

        time = sbd.col(X, self.time_col)
        min_date = sbd.min(time)
        max_date = sbd.max(time)

        splits = self.split(X)
        for i, (train_idx, test_idx) in enumerate(splits):
            train = sbd.col(sbd.select_rows(X, train_idx), self.time_col)
            test = sbd.col(sbd.select_rows(X, test_idx), self.time_col)

            # Calculate date ranges
            train_start = sbd.min(train)
            train_end = sbd.max(train)
            test_start = sbd.min(test)
            test_end = sbd.max(test)

            # inferred from splits but can be accessed by calling get_boundaries()
            gap_start = train_end
            gap_end = test_start
            gap_days_actual = (gap_end - gap_start).days

            # Convert to matplotlib dates
            train_duration = mdates.date2num(train_end) - mdates.date2num(train_start)
            gap_duration = mdates.date2num(gap_end) - mdates.date2num(gap_start)
            test_duration = mdates.date2num(test_end) - mdates.date2num(test_start)

            # Plot train bar
            ax.barh(
                i,
                width=train_duration,
                left=mdates.date2num(train_start),
                height=0.8,
                color=format["train_color"],
                alpha=format["alpha"],
                edgecolor="black",
                linewidth=1,
                label="Train" if i == 0 else "",
            )

            # Add train duration text
            train_mid = train_start + (train_end - train_start) / 2
            train_days_actual = (train_end - train_start).days
            ax.text(
                mdates.date2num(train_mid),
                i,
                f"{train_days_actual}d",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black",
            )

            # Plot gap bar (if gap exists)
            if gap_days_actual > 0:
                ax.barh(
                    i,
                    width=gap_duration,
                    left=mdates.date2num(gap_start),
                    height=0.8,
                    facecolor=format["gap_facecolor"],
                    edgecolor=format["gap_color"],
                    hatch=format["gap_hatch"],
                    linewidth=1,
                    label="Gap" if i == 0 else "",
                )
                # Add gap text
                gap_mid = gap_start + (gap_end - gap_start) / 2
                ax.text(
                    mdates.date2num(gap_mid),
                    i,
                    f"{gap_days_actual}d",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                )

            # Plot test bar
            ax.barh(
                i,
                width=test_duration,
                left=mdates.date2num(test_start),
                height=0.8,
                color=format["test_color"],
                alpha=format["alpha"],
                edgecolor="black",
                linewidth=1,
                label="Test" if i == 0 else "",
            )

            # Add test duration text
            test_mid = test_start + (test_end - test_start) / 2
            test_days_actual = (test_end - test_start).days
            ax.text(
                mdates.date2num(test_mid),
                i,
                f"{test_days_actual}d",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black",
            )

        # Formatting
        n_splits = self.get_n_splits(X)
        ax.set_yticks(range(n_splits))
        ax.set_yticklabels([f"Split {j}" for j in range(n_splits)])
        ax.set_ylim(-0.5, n_splits - 0.5)
        ax.set_xlim(
            mdates.date2num(min_date - pd.Timedelta(days=5)),
            mdates.date2num(max_date + pd.Timedelta(days=5)),
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Splits", fontsize=12)
        ax.set_title(
            f"Time Splits - Test:{self.test_frequency}, Gap: {self.gap_size}",
            fontsize=12,
            pad=10,
        )

        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        ax.tick_params(axis="both", labelsize=10)

        # Add legend
        ax.legend(fontsize=10, loc=format["loc"])

        fig.autofmt_xdate()

        return fig, ax
