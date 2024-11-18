# drop columns that contain all null values
import warnings

from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._on_each_column import RejectColumn, SingleColumnTransformer

__all__ = ["DropColumnIfNull"]


class DropColumnIfNull(SingleColumnTransformer):
    """Drop a single column if it contains only Null, NaN, or a mixture of null
    values. If at least one non-null value is found, the column is kept.


    Parameters
    ----------
    null_column_strategy : str, default="warn", "drop", "keep", "raise"
        If `warn`, columns that contain only null values are kept, but a warning
        is issued.
        If `drop`, null columns are dropped.
        If `keep`, null columns are kept as is, and no warning is raised.
        If `raise`, a RejectColumn exception is raised if a null column is detected.
    """

    def __init__(self, null_column_strategy="warn"):
        if null_column_strategy not in ["warn", "drop", "keep", "raise"]:
            raise ValueError(f"Unknown strategy {null_column_strategy}")
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
                return column
            if self.null_column_strategy == "drop":
                return []
            if self.null_column_strategy == "keep":
                return column
            if self.null_column_strategy == "raise":
                raise RejectColumn(
                    f"Column {sbd.name(column)} contains only null values."
                )
        return column
