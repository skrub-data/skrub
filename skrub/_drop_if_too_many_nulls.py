import numbers

from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._on_each_column import SingleColumnTransformer

__all__ = ["DropIfTooManyNulls"]


class DropIfTooManyNulls(SingleColumnTransformer):
    """Drop a single column if the fraction of Null or NaN values in the column
    is larger than the given threshold.

    If the threshold is set to `1.0`, the column is dropped if it contains only
    nulls or NaNs (this is the default value). If the threshold is set to `None`,
    all columns are kept. Otherwise, the column is dropped if the fraction
    of nulls is strictly larger than the threshold

    Parameters
    ----------
    threshold : float in range [0, 1], or None
        Threshold of null values past which the column is dropped.
    """

    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def fit_transform(self, column, y=None):
        """Fit the encoder and transform a column.

        Parameters
        ----------
            column : Pandas or Polars series. The input column to check.
            y : None. Ignored.

        Returns
        -------
            The input column, or an empty list if the column is selected to be
            dropped depending on the threshold.
        """
        del y

        if self.threshold is not None:
            if (
                not isinstance(self.threshold, numbers.Number)
                or not 0.0 <= self.threshold <= 1.0
            ):
                raise ValueError(
                    f"Threshold {self.threshold} is invalid. Threshold should be "
                    "a number in the range [0, 1]."
                )

        if self.threshold == 1.0:
            self.drop_ = sbd.is_all_null(column)
        elif self.threshold is None:
            self.drop_ = False
        else:
            # Count nulls
            null_count = sum(sbd.is_null(column))
            # No nulls found
            if null_count == 0:
                self.drop_ = False
            else:  # some nulls found, check if fraction > threshold
                self.drop_ = null_count / len(column) > self.threshold
        return self.transform(column)

    def transform(self, column):
        """Transform a column.

        Parameters:
        -----------
            column : Pandas or Polars series. The input column to check.

        Returns
        -------
        column
            The input column, or an empty list if the column is chosen to be
            dropped.
        """
        check_is_fitted(self)

        if self.drop_:
            return []
        return column
