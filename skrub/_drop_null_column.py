# drop columns that contain all null values
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._on_each_column import SingleColumnTransformer

__all__ = ["DropNullColumn"]


class DropNullColumn(SingleColumnTransformer):
    """Drop a single column if it contains only null values."""

    def fit_transform(self, column, y=None):
        """Fit the encoder and transform a column.

        Args:
            column : Pandas or Polars series. The input column to check.
            y : None. Ignored.

        Returns:
            The input column, or an empty list if the column contains only null values.
        """
        del y

        self.drop_ = sbd.is_all_null(column)

        return self.transform(column)

    def transform(self, column):
        """Transform a column.

        Args:
            column : Pandas or Polars series. The input column to check.

        Returns:
            The input column, or an empty list if the column contains only null values.
        """
        check_is_fitted(self)

        if self.drop_:
            return []
        return column
