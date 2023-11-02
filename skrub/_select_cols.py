from sklearn.base import BaseEstimator, TransformerMixin

from ._dataframe._namespace import get_df_namespace


def _check_columns(df, columns):
    """Check that provided columns exist in the dataframe and return them in a list.

    Checking this ourselves allows having the same exception for both pandas
    and polars dataframes.

    If `df` is not a dataframe (does not have a ``columns`` attribute), skip
    the check. As the transformers in this module are basically stateless,
    this allows getting an operational transformer without fit data; for
    example ``selector = SelectCols(["A", "B"]).fit(None)``, as the fit data is
    not used for anything else than this check.

    If ``columns`` is a ``str`` (a single column name), the return value wraps
    it in a list (of length 1).
    """
    if isinstance(columns, str):
        columns = [columns]
    columns = list(columns)
    if not hasattr(df, "columns"):
        return columns
    diff = set(columns) - set(df.columns)
    if not diff:
        return columns
    raise ValueError(
        f"The following columns were not found in the input DataFrame: {diff}"
    )


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
    ValueError: The following columns were not found in the input DataFrame: {'X'}
    """

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
        _check_columns(X, self.cols)
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
        cols = _check_columns(X, self.cols)
        namespace, _ = get_df_namespace(X)
        return namespace.select(X, cols)


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
    ValueError: The following columns were not found in the input DataFrame: {'X'}
    """

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
        _check_columns(X, self.cols)
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
        cols = _check_columns(X, self.cols)
        namespace, _ = get_df_namespace(X)
        return namespace.select(X, [c for c in X.columns if c not in cols])
