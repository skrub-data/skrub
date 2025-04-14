import numbers

from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._on_each_column import SingleColumnTransformer

__all__ = ["DropUninformative"]


class DropUninformative(SingleColumnTransformer):
    """Drop column if it is found to be uninformative according to various criteria.

    A column is considered to be "uninformative" if one or more of the following
    issues are found:
    - The fraction of missing values is larger than a certain fraction (by default,
    all values must be null for the column to be dropped).
    - The column includes only one unique value (the column is constant). Missing
    values are considered a separate value.
    - The number of unique values in the column is equal to the length of the column.

    Parameters
    ----------
    constant_column: bool, default=True
        If True, drop the column if it contains only one unique value. Missing values
        count as one additional distinct value.
    column_is_id: bool, default=False
        If True, drop the column if all values are distinct. Missing values count as
        one additional distinct value.
    null_fraction_threshold: float or None, default=1.0
        Drop columns with a fraction of missing values larger than threshold. If None,
        keep the column even if all its values are missing.

    """

    def __init__(
        self,
        constant_column=True,
        column_is_id=False,
        null_fraction_threshold=1.0,
    ):
        self.constant_column = constant_column
        self.column_is_id = column_is_id
        self.null_fraction_threshold = null_fraction_threshold

    def _check_params(self):
        if self.constant_column not in [True, False]:
            raise ValueError(
                "constant_column must be in [True, False], found"
                f" {self.constant_column}."
            )
        if self.column_is_id not in [True, False]:
            raise ValueError(
                f"column_is_id must be in [True, False], found {self.column_is_id}."
            )

        if self.null_fraction_threshold is not None:
            if (
                not isinstance(self.null_fraction_threshold, numbers.Number)
                or not 0.0 <= self.null_fraction_threshold <= 1.0
            ):
                raise ValueError(
                    f"Threshold {self.null_fraction_threshold} is invalid. Threshold"
                    " should be a number in the range [0, 1]."
                )

    def _drop_if_too_many_nulls(self, column):
        if self.null_fraction_threshold == 1.0:
            return sbd.is_all_null(column)
        # No nulls found, or no threshold
        if self.null_count == 0 or self.null_fraction_threshold is None:
            return False
        return self.null_count / len(column) > self.null_fraction_threshold

    def _drop_if_constant(self, column):
        if self.constant_column:
            if (sbd.n_unique(column) == 1) and (sum(sbd.is_null(column)) == 0):
                return True
        return False

    def _drop_if_id(self, column):
        if self.column_is_id:
            n_unique = sbd.n_unique(column)
            if self.null_count > 0:
                return False
            if n_unique == len(column):
                return True
        return False

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

        self._check_params()

        # Count nulls
        self.null_count = sum(sbd.is_null(column))

        self.drop_ = any(
            [
                self._drop_if_too_many_nulls(column),
                self._drop_if_constant(column),
                self._drop_if_id(column),
            ]
        )

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
