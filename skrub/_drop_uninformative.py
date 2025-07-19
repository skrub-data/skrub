import numbers

from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from ._apply_to_cols import SingleColumnTransformer

__all__ = ["DropUninformative"]


class DropUninformative(SingleColumnTransformer):
    """Drop column if it is found to be uninformative according to various criteria.

    Columns are considered "uninformative" if the fraction of missing values is larger
    than a threshold, if they contain one unique value, or if all values are unique.

    Parameters
    ----------
    drop_if_constant : bool, default=False
        If True, drop the column if it contains only one unique value. Missing values
        count as one additional distinct value.

    drop_if_unique : bool, default=False
        If True, drop the column if all values are distinct. Missing values count as
        one additional distinct value. Numeric columns are never dropped. This may
        lead to dropping columns that contain free-flowing text.

    drop_null_fraction : float or None, default=1.0
        Drop columns with a fraction of missing values larger than threshold. If None,
        keep the column even if all its values are missing.

    Notes
    -----
    A column is considered to be "uninformative" if one or more of the following
    issues are found:

    - The fraction of missing values is larger than a certain fraction (by default,
      all values must be null for the column to be dropped).
    - The column includes only one unique value (the column is constant). Missing
      values are considered a separate value.
    - The number of unique values in the column is equal to the length of the
      column, i.e., all values are unique. This is only considered for non-numeric
      columns. Missing values are considered a separate value. Note that this
      may lead to dropping columns that contain free-flowing text.

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

    >>> df = pd.DataFrame({"col1": [1, 2, None], "col2": ["const", "const", "const"]})
    >>> du = DropUninformative(drop_if_constant=True, drop_null_fraction=0.1)
    >>> du.fit_transform(df["col1"])
    []
    >>> du.fit_transform(df["col2"])
    []

    Finally, it is possible to set ``drop_if_unique`` to ``True`` in order to drop
    string columns that contain all distinct values:

    >>> df = pd.DataFrame({"col1": ["A", "B", "C"]})
    >>> du = DropUninformative(drop_if_unique=True)
    >>> du.fit_transform(df["col1"])
    []
    """

    def __init__(
        self,
        drop_if_constant=False,
        drop_if_unique=False,
        drop_null_fraction=1.0,
    ):
        self.drop_if_constant = drop_if_constant
        self.drop_if_unique = drop_if_unique
        self.drop_null_fraction = drop_null_fraction

    def _check_params(self):
        if not isinstance(self.drop_if_constant, bool):
            raise TypeError(
                f"drop_if_constant must be boolean, found {self.drop_if_constant}."
            )
        if not isinstance(self.drop_if_unique, bool):
            raise TypeError(
                f"drop_if_unique must be boolean, found {self.drop_if_unique}."
            )

        if self.drop_null_fraction is not None:
            if (
                not isinstance(self.drop_null_fraction, numbers.Number)
                or not 0.0 <= self.drop_null_fraction <= 1.0
            ):
                raise ValueError(
                    f"Threshold {self.drop_null_fraction} is invalid. Threshold"
                    " should be a number in the range [0, 1], or None."
                )

    def _drop_if_too_many_nulls(self, column):
        if self.drop_null_fraction == 1.0:
            return self._null_count == len(column)
        # No nulls found, or no threshold
        if self._null_count == 0 or self.drop_null_fraction is None:
            return False
        return self._null_count / len(column) > self.drop_null_fraction

    def _drop_if_constant(self, column):
        if self.drop_if_constant:
            if (sbd.n_unique(column) == 1) and (self._null_count == 0):
                return True
        return False

    def _drop_if_unique(self, column):
        if self.drop_if_unique and not sbd.is_numeric(column):
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
                self._drop_if_unique,
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
