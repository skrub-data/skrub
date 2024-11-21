# drop columns that contain all null values
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._on_each_column import SingleColumnTransformer

__all__ = ["DropColumnIfNull"]


class DropColumnIfNull(SingleColumnTransformer):
    """Drop a single column if the fraction of Null or NaN values in the column
    is larger than the given threshold.

    By default, the threshold is set to `1.0`, so only columns that contain only
    nulls or NaNs are dropped. Set the threshold to `None` to keep all columns.

    Parameters
    ----------
    threshold : float, or None
        Threshold of null values past which the column is dropped.
    """

    def __init__(self, threshold: float | None = 1.0):
        assert (
            0.0 < threshold < 1.0
        ) or threshold is None, "Invalid value for the threshold."

        self.threshold = threshold

    def fit_transform(self, column, y=None):
        """Fit the encoder and transform a column.

        Parameters
        ----------
            column : Pandas or Polars series. The input column to check.
            y : None. Ignored.

        Returns
        -------
            The input column, or an empty list if the column contains only null values.
        """
        del y

        if self.threshold == 1.0:
            self.drop_ = sbd.is_all_null(column)
        elif self.threshold is None:
            self.drop_ = False
        else:
            n_count = sum(sbd.is_null(column))
            if n_count / len(column) > self.threshold:
                self.drop_ = True
            else:
                self.drop_ = False
        return self.transform(column)

    def transform(self, column):
        """Transform a column.

        Parameters:
        -----------
            column : Pandas or Polars series. The input column to check.

        Returns
        -------
        column
            The input column, or an empty list if the column contains only null values.
        """
        check_is_fitted(self)

        if self.drop_:
            return []
        return column
