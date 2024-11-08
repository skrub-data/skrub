# drop columns that contain all null values
import warnings

from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._on_each_column import RejectColumn, SingleColumnTransformer

__all__ = ["DropColumnIfNull"]


class DropColumnIfNull(SingleColumnTransformer):
    """Drop a single column if it contains only Null, NaN, or a mixture of null
    values. If at least one non-null value is found, the column is kept."""

    def __init__(self, null_column_strategy="warn"):
        self.null_column_strategy = null_column_strategy

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

        self.drop_ = sbd.is_all_null(column)

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
            if self.null_column_strategy == "warn":
                warnings.warn(
                    f"Column {sbd.name(column)} contains only null values.",
                    UserWarning,
                    stacklevel=2,
                )
                return []
            if self.null_column_strategy == "drop":
                return []
            if self.null_column_strategy == "ignore":
                return column
            if self.null_column_strategy == "raise":
                raise RejectColumn(
                    f"Column {sbd.name(column)} contains only null values."
                )
        return column
