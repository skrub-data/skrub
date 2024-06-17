from sklearn.base import BaseEstimator, TransformerMixin

from . import _selectors as s
from ._on_each_column import SingleColumnTransformer


class SelectCols(TransformerMixin, BaseEstimator):
    """Select a subset of a DataFrame's columns.

    A ``ValueError`` is raised if any of the provided column names are not in
    the dataframe.

    Accepts :obj:`pandas.DataFrame` and :obj:`polars.DataFrame` inputs.

    Parameters
    ----------
    cols : list of str or str
        The columns to select. A single column name can be passed as a ``str``:
        ``"col_name"`` is the same as ``["col_name"]``.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub import SelectCols
    >>> df = pd.DataFrame({"A": [1, 2], "B": [10, 20], "C": ["x", "y"]})
    >>> df
       A   B  C
    0  1  10  x
    1  2  20  y
    >>> SelectCols(["C", "A"]).fit_transform(df)
       C  A
    0  x  1
    1  y  2
    >>> SelectCols(["X", "A"]).fit_transform(df)
    Traceback (most recent call last):
        ...
    ValueError: The following columns are requested for selection but missing from dataframe: ['X']
    """  # noqa: E501

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        """Fit the transformer.

        Parameters
        ----------
        X : DataFrame or None
            If `X` is a DataFrame, the transformer checks that all the column
            names provided in ``self.cols`` can be found in `X`.

        y : None
            Unused.

        Returns
        -------
        SelectCols
            The transformer itself.
        """
        self._columns = s.make_selector(self.cols).expand(X)
        return self

    def transform(self, X):
        """Transform a dataframe by selecting columns.

        Parameters
        ----------
        X : DataFrame
            The DataFrame on which to apply the selection.

        Returns
        -------
        DataFrame
            The input DataFrame ``X`` after selecting only the columns listed
            in ``self.cols`` (in the provided order).
        """
        return s.select(X, self._columns)


class DropCols(TransformerMixin, BaseEstimator):
    """Drop a subset of a DataFrame's columns.

    The other columns are kept in their original order. A ``ValueError`` is
    raised if any of the provided column names are not in the dataframe.

    Accepts :obj:`pandas.DataFrame` and :obj:`polars.DataFrame` inputs.

    Parameters
    ----------
    cols : list of str or str
        The columns to drop. A single column name can be passed as a ``str``:
        ``"col_name"`` is the same as ``["col_name"]``.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub import DropCols
    >>> df = pd.DataFrame({"A": [1, 2], "B": [10, 20], "C": ["x", "y"]})
    >>> df
       A   B  C
    0  1  10  x
    1  2  20  y
    >>> DropCols(["A", "C"]).fit_transform(df)
        B
    0  10
    1  20
    >>> DropCols(["X"]).fit_transform(df)
    Traceback (most recent call last):
        ...
    ValueError: The following columns are requested for selection but missing from dataframe: ['X']
    """  # noqa: E501

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        """Fit the transformer.

        Parameters
        ----------
        X : DataFrame or None
            If `X` is a DataFrame, the transformer checks that all the column
            names provided in ``self.cols`` can be found in `X`.

        y : None
            Unused.

        Returns
        -------
        DropCols
            The transformer itself.
        """
        self._columns = s.make_selector(self.cols).expand(X)
        return self

    def transform(self, X):
        """Transform a dataframe by dropping columns.

        Parameters
        ----------
        X : DataFrame
            The DataFrame on which to apply the selection.

        Returns
        -------
        DataFrame
            The input DataFrame ``X`` after dropping the columns listed in
            ``self.cols``.
        """
        return s.select(X, ~s.make_selector(self._columns))


class Drop(SingleColumnTransformer):
    def fit_transform(self, column, y=None):
        return []

    def transform(self, column):
        return []
