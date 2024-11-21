# drop columns that contain all null values
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._on_each_column import SingleColumnTransformer

__all__ = ["DropColumnIfNull"]


class DropColumnIfNull(SingleColumnTransformer):
    """Drop a single column if it contains only Null, NaN, or a mixture of null
    values. If at least one non-null value is found, the column is kept."""

    def __init__(self, threshold: float = 1.0):
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
