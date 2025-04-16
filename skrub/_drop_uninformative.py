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
    - The number of unique values in the column is equal to the length of the
    column.

    Parameters
    ----------
    drop_if_constant : bool, default=False
        If True, drop the column if it contains only one unique value. Missing values
        count as one additional distinct value.

    drop_if_id : bool, default=False
        If True, drop the column if all values are distinct. Missing values count as
        one additional distinct value. Numeric columns are never dropped.

    null_fraction_threshold : float or None, default=1.0
        Drop columns with a fraction of missing values larger than threshold. If None,
        keep the column even if all its values are missing.

    Examples
    --------
    >>> from skrub import DropUninformative
    >>> import pandas as pd
    >>> df = pd.DataFrame({"col1": [None, None, None]})

    By default, only null columns are dropped:

    >>> du = DropUninformative()
    >>> du.fit_transform(df["col1"])
    []

    It is also possible to drop constant columns, or specify a lower null fraction
    threshold:

    >>> df = pd.DataFrame({"col1": [1,2,None], "col2": ["const", "const", "const"]})
    >>> du = DropUninformative(drop_if_constant=True, null_fraction_threshold=0.1)
    >>> du.fit_transform(df["col1"])
    []
    >>> du.fit_transform(df["col2"])
    []

    Finally, it is possible to set ``drop_if_id`` to ``True`` in order to drop
    string columns that contain all distinct values:

    >>> df = pd.DataFrame({"col1": ["A", "B", "C"]})
    >>> du = DropUninformative(drop_if_id=True)
    >>> du.fit_transform(df["col1"])
    []
    """

    def __init__(
        self,
        drop_if_constant=True,
        drop_if_id=False,
        null_fraction_threshold=1.0,
    ):
        self.drop_if_constant = drop_if_constant
        self.drop_if_id = drop_if_id
        self.null_fraction_threshold = null_fraction_threshold

    def _check_params(self):
        if not isinstance(self.drop_if_constant, bool):
            raise TypeError(
                f"drop_if_constant must be boolean, found {self.drop_if_constant}."
            )
        if not isinstance(self.drop_if_id, bool):
            raise TypeError(f"drop_if_id must be boolean, found {self.drop_if_id}.")

        if self.null_fraction_threshold is not None:
            if (
                not isinstance(self.null_fraction_threshold, numbers.Number)
                or not 0.0 <= self.null_fraction_threshold <= 1.0
            ):
                raise ValueError(
                    f"Threshold {self.null_fraction_threshold} is invalid. Threshold"
                    " should be a number in the range [0, 1], or None."
                )

    def _drop_if_too_many_nulls(self, column):
        if self.null_fraction_threshold == 1.0:
            return sbd.is_all_null(column)
        # No nulls found, or no threshold
        if self._null_count == 0 or self.null_fraction_threshold is None:
            return False
        return self._null_count / len(column) > self.null_fraction_threshold

    def _drop_if_constant(self, column):
        if self.drop_if_constant:
            if (sbd.n_unique(column) == 1) and (not sbd.has_nulls(column)):
                return True
        return False

    def _drop_if_id(self, column):
        if self.drop_if_id and not sbd.is_numeric(column):
            n_unique = sbd.n_unique(column)
            if self._null_count > 0:
                return False
            if n_unique == len(column):
                return True
        return False

    def fit_transform(self, column, y=None):
        """Fit the encoder and transform a column.

        Parameters
        ----------
        column : Pandas or Polars series
            The input column to check.
        y : None
            Ignored.

        Returns
        -------
        column
            The input column, or an empty list if the column is chosen to be
            dropped.
        """
        del y

        self._check_params()

        # Count nulls
        self._null_count = sum(sbd.is_null(column))

        self.drop_ = any(
            check(column)
            for check in [
                self._drop_if_too_many_nulls,
                self._drop_if_constant,
                self._drop_if_id,
            ]
        )

        return self.transform(column)

    def transform(self, column):
        """Transform a column.

        Parameters
        -----------
        column : Pandas or Polars series
            The input column to check.

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
