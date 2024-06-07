from . import _dataframe as sbd
from ._on_each_column import RejectColumn, SingleColumnTransformer

__all__ = ["ToStr"]


class ToStr(SingleColumnTransformer):
    """
    Convert a column to strings.

    A numeric, datetime or categorical column is rejected with a
    ``RejectColumn`` exception. This is to avoid accidentally converting a
    column that already has a more informative dtype.

    Any other column is converted to a column of strings. Null values are
    preserved, i.e. will still be nulls in the output.

    For pandas, the output always has the ``object`` dtype (the old way of
    representing strings in pandas), not the more recent ``StringDtype``
    extension dtype, and null values are represented by ``np.nan``. This is due
    to the fact that scikit-learn estimators and encoders do not yet handle
    pandas extension dtypes, especially in the presence of missing values
    (represented by ``pd.NA``).

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub._to_str import ToStr
    >>> to_str = ToStr()

    A non-string column is converted to strings:

    >>> s = pd.Series([('one', 1), ('two', 2), None])
    >>> s
    0    (one, 1)
    1    (two, 2)
    2        None
    dtype: object
    >>> _[0]
    ('one', 1)
    >>> to_str.fit_transform(s)
    0    ('one', 1)
    1    ('two', 2)
    2           NaN
    dtype: object
    >>> _[0]
    "('one', 1)"

    A string column with the extension dtype ``StringDtype`` is converted to the
    ``object`` dtype, and ``pd.NA`` is converted to ``np.nan``:

    >>> s = pd.Series(['one', 'two', None], dtype='string')
    >>> s
    0     one
    1     two
    2    <NA>
    dtype: string
    >>> to_str.fit_transform(s)
    0    one
    1    two
    2    NaN
    dtype: object

    In ``object`` columns, ``pd.NA`` and other null values are also replaced by
    ``np.nan``:

    >>> s = pd.Series([{'city': 'Paris'}, None, pd.NaT, pd.NA])
    >>> s
    0    {'city': 'Paris'}
    1                 None
    2                  NaT
    3                 <NA>
    dtype: object
    >>> to_str.fit_transform(s)
    0    {'city': 'Paris'}
    1                  NaN
    2                  NaN
    3                  NaN
    dtype: object

    A column that already contain strings, has the dtype ``object`` and no
    missing values is passed through:

    >>> s = pd.Series(['one', 'two'])
    >>> to_str.fit_transform(s) is s
    True

    For other pandas columns, a copy or a modified copy is returned.

    A numeric, datetime or categorical column is rejected:

    >>> to_str.fit_transform(pd.Series([1.1, 2.2], name='s'))
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Refusing to convert 's' with dtype 'float64' to strings.
    >>> to_str.fit_transform(pd.Series(['a', 'b'], name='s', dtype='category'))
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Refusing to convert 's' with dtype 'category' to strings.
    >>> to_str.fit_transform(pd.to_datetime(pd.Series(['2020-02-02'])))
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Refusing to convert None with dtype 'datetime64[ns]' to strings.

    However, once a column has been accepted, the output of ``transform`` will
    always be strings:

    >>> to_str.fit(pd.Series(['a', 'b']))
    ToStr()
    >>> to_str.transform(pd.Series([1.1, 2.2]))
    0    1.1
    1    2.2
    dtype: object
    >>> _[0]
    '1.1'

    For polars, a string column is always passed through:

    >>> import pytest
    >>> pl = pytest.importorskip('polars')
    >>> s = pl.Series('s', ['one', 'two', None])
    >>> to_str.fit_transform(s) is s
    True

    A column that is neither String, categorical, numeric or datetime is converted:

    >>> s = pl.Series('s', [{'name':'one', 'value': 1}, {'name': 'two', 'value': 2}])
    >>> s
    shape: (2,)
    Series: 's' [struct[2]]
    [
        {"one",1}
        {"two",2}
    ]
    >>> to_str.fit_transform(s)
    shape: (2,)
    Series: 's' [str]
    [
        "{"one",1}"
        "{"two",2}"
    ]

    The conversion also works for Object columns. However converting Object
    columns may be much slower than other columns.

    >>> s = pl.Series([to_str, 'one', 3.3])
    >>> s
    shape: (3,)
    Series: '' [o][object]
    [
        ToStr()
        one
        3.3
    ]
    >>> to_str.fit_transform(s)
    shape: (3,)
    Series: '' [str]
    [
        "ToStr()"
        "one"
        "3.3"
    ]

    Categorical and Enum columns, numeric, Date and Datetime columns are rejected:

    >>> to_str.fit_transform(pl.Series('s', ['a', 'b'], dtype=pl.Enum(['a', 'b'])))
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Refusing to convert 's' with dtype 'Enum(categories=['a', 'b'])' to strings.
    >>> to_str.fit_transform(pl.Series('s', ['2020-02-01']).cast(pl.Date))
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Refusing to convert 's' with dtype 'Date' to strings.
    """  # noqa: E501

    def fit_transform(self, column, y=None):
        del y
        if (
            sbd.is_numeric(column)
            or sbd.is_any_date(column)
            or sbd.is_categorical(column)
        ):
            raise RejectColumn(
                f"Refusing to convert {sbd.name(column)!r} "
                f"with dtype '{sbd.dtype(column)}' to strings."
            )
        return self.transform(column)

    def transform(self, column):
        return sbd.to_string(column)
