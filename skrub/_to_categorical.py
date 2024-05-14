from . import _dataframe as sbd
from ._on_each_column import RejectColumn, SingleColumnTransformer


class ToCategorical(SingleColumnTransformer):
    """
    Convert a string column to Categorical dtype.

    A pandas columns with dtype ``string`` or ``object`` containing strings, or
    a polars column with dtype ``String``, is converted to a categorical
    column. Any other type of column is rejected with a ``RejectColumn``
    exception.

    The output of ``transform`` also always has a Categorical dtype. The categories
    are not necessarily the same across different calls to ``transform``. Indeed,
    scikit-learn estimators do not inspect the dtype's categories but the actual
    values. Converting to a Categorical itself is therefore just a way to mark a
    column and indicate to downstream estimators that this column should be treated
    as categorical. Ensuring they are encoded consistently, handling unseen
    categories at test time, etc. is the responsibility of encoders such as
    ``OneHotEncoder`` and ``LabelEncoder``, or of estimators that handle categories
    themselves such as ``HistGradientBoostingRegressor``.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub._to_categorical import ToCategorical

    A string column is converted to a categorical column.

    >>> s = pd.Series(['one', 'two', None], name='c')
    >>> to_cat = ToCategorical()
    >>> to_cat.fit_transform(s)
    0    one
    1    two
    2    NaN
    Name: c, dtype: category
    Categories (2, object): ['one', 'two']

    The dtypes (the list of categories) of the outputs of ``transform`` may
    vary. This transformer only ensures the dtype is Categorical to mark the
    column as such for downstream encoders which will perform the actual
    encoding.

    >>> s = pd.Series(['four', 'five'], name='c')
    >>> to_cat.transform(s)
    0    four
    1    five
    Name: c, dtype: category
    Categories (2, object): ['five', 'four']

    Columns that are not strings are rejected.

    >>> to_cat.fit_transform(pd.Series([1.1, 2.2], name='c'))
    Traceback (most recent call last):
        ...
    skrub._exceptions.RejectColumn: Column 'c' does not contain strings.

    In particular, columns that are already categorical are rejected.

    >>> to_cat.fit_transform(pd.Series(['one', 'two'], name='c', dtype='category'))
    Traceback (most recent call last):
        ...
    skrub._exceptions.RejectColumn: Column 'c' does not contain strings.

    ``object`` columns that do not contain only strings are also rejected.

    >>> s = pd.Series(['one', 1], name='c')
    >>> to_cat.fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._exceptions.RejectColumn: Column 'c' does not contain strings.

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
    Categories (2, string): [cat A, cat B]
    >>> _.cat.categories.dtype
    string[python]

    See ``CleanCategories`` to convert categories to object dtypes or
    ``PandasStringDtypetoObject`` to convert strings to object dtypes.

    Polars columns are converted to the ``Categorical`` dtype (not ``Enum``). As
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
    """

    def fit_transform(self, column):
        if not sbd.is_string(column):
            raise RejectColumn(f"Column {sbd.name(column)!r} does not contain strings.")
        return sbd.to_categorical(column)

    def transform(self, column):
        return sbd.to_categorical(column)
