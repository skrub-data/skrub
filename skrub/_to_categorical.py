from . import _dataframe as sbd
from ._single_column_transformer import RejectColumn, SingleColumnTransformer

__all__ = ["ToCategorical"]


class ToCategorical(SingleColumnTransformer):
    """
    Convert a string column to Categorical dtype.

    This transformer ensures that a given string or categorical column has
    Categorical dtype so that it is treated as categorical
    by downstream transformers and learners.

    Parameters
    ----------
    accept_numeric : str, default="int"
        How to handle numeric columns. If "int", will convert integer
        columns to categorical. If "all", both float and integer columns
        will be accepted. If `None`, no numeric
        columns will be accepted.

    Notes
    -----
    The main benefit of converting columns to categorical is that categorical
    columns can be recognized by
    scikit-learn's ``HistGradientBoostingRegressor`` and
    ``HistGradientBoostingClassifier`` with their
    ``categorical_features='from_dtype'`` option. This transformer is therefore
    particularly useful as the ``low_cardinality_transformer`` parameter of the
    ``TableVectorizer`` when combined with one of those supervised learners.

    A pandas column with dtype ``string`` or ``object`` containing strings, or
    a polars column with dtype ``String``, is converted to a categorical
    column. Categorical columns are passed through.

    If ``accept_numeric`` is set to ``"all"``, then both integer and float
    columns are accepted and converted to categorical. If it is set to ``"int"``,
    then only integer columns are accepted. The default value is ``"int"``.

    Any other type of column is rejected by raising a ``RejectColumn``
    exception. **Note:** the ``TableVectorizer`` only sends string or
    categorical columns to its ``low_cardinality_transformer``, regardless
    of the inputted value of ``accept_numeric``. Therefore it is
    always safe to use a ``ToCategorical`` instance as the
    ``low_cardinality_transformer``.

    The output of ``transform`` also always has a Categorical dtype. The categories
    are not necessarily the same across different calls to ``transform``. Indeed,
    scikit-learn estimators do not inspect the dtype's categories but the actual
    values. Converting to a Categorical is therefore just a way to mark a
    column and indicate to downstream estimators that this column should be treated
    as categorical. Ensuring they are encoded consistently, handling unseen
    categories at test time, etc. is the responsibility of encoders such as
    ``OneHotEncoder`` and ``LabelEncoder``, or of estimators that handle categories
    themselves such as ``HistGradientBoostingRegressor``.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub import ToCategorical

    A string column is converted to a categorical column.

    >>> s = pd.Series(['one', 'two', None], name='c')
    >>> s # doctest: +SKIP
    0     one
    1     two
    2     ...
    Name: c, dtype: ...
    >>> to_cat = ToCategorical()
    >>> to_cat.fit_transform(s) # doctest: +SKIP
    0    one
    1    two
    2    ...
    Name: c, dtype: ...
    Categories (2, ...): ['one', 'two']

    The dtypes (the list of categories) of the outputs of ``transform`` may
    vary. This transformer only ensures the dtype is Categorical to mark the
    column as such for downstream encoders which will perform the actual
    encoding.

    >>> s = pd.Series(['four', 'five'], name='c')
    >>> to_cat.transform(s)
    0    four
    1    five
    Name: c, dtype: category
    Categories (2, ...): ['five', 'four']

    Columns that already have a Categorical dtype are passed through:

    >>> s = pd.Series(['one', 'two'], name='c', dtype='category')
    >>> to_cat.fit_transform(s) is s
    True

    Columns that are not strings nor categorical are rejected:

    >>> to_cat.fit_transform(pd.Series([1.1, 2.2], name='c'))
    Traceback (most recent call last):
        ...
    skrub.core.RejectColumn: Column 'c' does not contain strings or accepted...

    Unless ``accept_numeric`` is set to ``"int"`` or ``"all"``, in which case integer
    columns are accepted in the former case, and both integer and float columns are
    accepted in the latter:

    >>> to_cat = ToCategorical(accept_numeric="all")
    >>> to_cat.fit_transform(pd.Series([1.1, 2.2], name='c'))
    0    1.1
    1    2.2
    Name: c, dtype: category
    Categories (2, float64): [1.1, 2.2]

    ``object`` columns that do not contain only strings are also rejected:

    >>> s = pd.Series(['one', 1], name='c')
    >>> to_cat.fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub.core.RejectColumn: Column 'c' does not contain strings or accepted...

    No special handling of ``StringDtype`` vs ``object`` columns is done, the
    behavior is the same as ``pd.astype('category')``: if the input uses the
    extension dtype, the categories of the output will, too.

    >>> s = pd.Series(['cat A', 'cat B', None], name='c', dtype='string')
    >>> s
    0    cat A
    1    cat B
    2     <NA>
    Name: c, dtype: string
    >>> to_cat.fit_transform(s)
    0    cat A
    1    cat B
    2     <NA>
    Name: c, dtype: category
    Categories (2, string): [...]
    >>> _.cat.categories.dtype # doctest: +SKIP

    Polars string columns are converted to the ``Categorical`` dtype (not ``Enum``). As
    for pandas, categories may vary across calls to ``transform``.

    >>> import pytest
    >>> pl = pytest.importorskip("polars")
    >>> s = pl.Series('c', ['one', 'two', None])
    >>> to_cat.fit_transform(s)
    shape: (3,)
    Series: 'c' [cat]
    [
        "one"
        "two"
        null
    ]

    Polars Categorical or Enum columns are passed through:

    >>> s = pl.Series('c', ['one', 'two'], dtype=pl.Enum(['one', 'two', 'three']))
    >>> s
    shape: (2,)
    Series: 'c' [enum]
    [
        "one"
        "two"
    ]
    >>> to_cat.fit_transform(s) is s
    True
    """

    def __init__(self, accept_numeric="int"):
        self.accept_numeric = accept_numeric
        super().__init__()

    def fit_transform(self, column, y=None):
        """Fit the encoder and transform a column.

        Parameters
        ----------
        column : pandas or polars Series

        y : None
            Ignored.

        Returns
        -------
        transformed : pandas or polars Series
            The input transformed to Categorical.
        """
        self.all_outputs_ = [sbd.name(column)]

        if sbd.is_categorical(column):
            return column
        if (
            sbd.is_string(column)
            or sbd.is_integer(column)
            and self.accept_numeric in ("int", "all")
            or sbd.is_float(column)
            and self.accept_numeric == "all"
        ):
            return sbd.to_categorical(column)
        raise RejectColumn(
            f"Column {sbd.name(column)!r} does not contain strings "
            "or accepted numeric types (if accept_numeric is 'int' or 'all')."
        )

    def transform(self, column):
        """Transform a column.

        Parameters
        ----------
        column : pandas or polars Series
            The input to transform.

        Returns
        -------
        transformed : pandas or polars Series
            The input transformed to Categorical.
        """
        if sbd.is_categorical(column):
            return column
        return sbd.to_categorical(column)
