from . import _dataframe as sbd
from ._dispatch import dispatch
from ._on_each_column import RejectColumn, SingleColumnTransformer

__all__ = ["CleanNullStrings"]

# Taken from pandas.io.parsers (version 1.1.4)
STR_NA_VALUES = [
    "null",
    "",
    "1.#QNAN",
    "#NA",
    "nan",
    "#N/A N/A",
    "-1.#QNAN",
    "<NA>",
    "-1.#IND",
    "-nan",
    "n/a",
    "-NaN",
    "1.#IND",
    "NULL",
    "NA",
    "N/A",
    "#N/A",
    "NaN",
    "?",
    "...",
]


@dispatch
def _trim_whitespace_only(col):
    raise NotImplementedError()


@_trim_whitespace_only.specialize("pandas", argument_type="Column")
def _trim_whitespace_only_pandas(col):
    return col.replace(r"^\s*$", "", regex=True)


@_trim_whitespace_only.specialize("polars", argument_type="Column")
def _trim_whitespace_only_polars(col):
    assert sbd.is_string(col), col.dtype
    return col.str.replace(r"^\s*$", "", literal=False)


class CleanNullStrings(SingleColumnTransformer):
    """Replace strings used to represent missing values with actual null values.

    For pandas, columns with dtypes ``object`` and ``string`` are considered;
    for polars, columns with dtype ``String``. (Note that in pandas ``object``
    is the default ``dtype`` to represent strings, ``string`` aka
    ``StringDtype()`` is an extension dtype used only if requested explicitly,
    which is why we also handle pandas ``object`` columns here.)

    See ``STR_NA_VALUES`` in this module for the full list of values considered
    as null.

    Examples
    --------

    >>> import pandas as pd
    >>> from skrub._clean_null_strings import CleanNullStrings
    >>> cleaner = CleanNullStrings()

    The null value depends on the input. If the input is a pandas ``object``
    column, ``None`` is used as the null value and the output is an ``object``
    column:

    >>> s = pd.Series(['one', 'N/A', '    ', True], name='s')
    >>> s
    0     one
    1     N/A
    2
    3    True
    Name: s, dtype: object
    >>> s.isna()
    0    False
    1    False
    2    False
    3    False
    Name: s, dtype: bool
    >>> (out := cleaner.fit_transform(s))
    0     one
    1    None
    2    None
    3    True
    Name: s, dtype: object
    >>> out.isna()
    0    False
    1     True
    2     True
    3    False
    Name: s, dtype: bool

    Non-string values and strings that do not represent missing values are left
    unchanged. In particular, non-string values in ``object`` columns are not
    cast to strings:

    >>> out[3], type(out[3])
    (True, <class 'bool'>)

    If the input uses the pandas string extension dtype, the null value is
    ``pd.NA`` and the output will have the same dtype as the input:

    >>> s = pd.Series(['one', 'N/A', ' ', 'two', pd.NA], name='s', dtype='string')
    >>> s
    0     one
    1     N/A
    2
    3     two
    4    <NA>
    Name: s, dtype: string
    >>> s.isna()
    0    False
    1    False
    2    False
    3    False
    4     True
    Name: s, dtype: bool
    >>> cleaner.fit_transform(s)
    0     one
    1    <NA>
    2    <NA>
    3     two
    4    <NA>
    Name: s, dtype: string
    >>> _.isna()
    0    False
    1     True
    2     True
    3    False
    4     True
    Name: s, dtype: bool

    No attempt is made to cast columns to a better type than ``object`` or
    ``string`` if it becomes possible after cleaning. This is handled by other
    transformers further down the ``skrub`` preprocessing pipeline, such as
    ``ToNumeric`` or ``ToDatetime``.

    >>> s = pd.Series(['1.1', '2.2', 'NaN', 'nan'], name='s', dtype='string')
    >>> cleaner.fit_transform(s)
    0     1.1
    1     2.2
    2    <NA>
    3    <NA>
    Name: s, dtype: string

    >>> s = pd.Series([1.1, 2.2, '<NA>', 4.4], name='s')
    >>> cleaner.fit_transform(s)
    0     1.1
    1     2.2
    2    None
    3     4.4
    Name: s, dtype: object

    In both examples above, the column can be converted to numbers by
    ``ToFloat`` (only) after being cleaned by ``CleanNullStrings``:

    >>> from skrub._to_float32 import ToFloat32
    >>> ToFloat32().fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Could not convert column 's' to numbers.
    >>> ToFloat32().fit_transform(cleaner.fit_transform(s))
    0    1.1
    1    2.2
    2    NaN
    3    4.4
    Name: s, dtype: float32

    Columns that are do not have ``object`` or ``string`` as their ``dtype``
    are rejected:

    >>> s = pd.Series([1.1, None], name='s')
    >>> cleaner.fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Column 's' does not contain strings.

    In particular, Categorical columns, although they contain strings, do not
    have the ``string`` or ``object`` ``dtype``:

    >>> s = pd.Series(['a', ''], dtype='category')
    >>> cleaner.fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Column None does not contain strings.

    Note however that ``object`` columns are accepted even if they do not
    contain any strings. They will not be modified but they will still be
    recorded as having been handled by this transformer:

    >>> s = pd.Series([True, False, None])
    >>> s
    0     True
    1    False
    2     None
    dtype: object
    >>> cleaner.fit_transform(s)
    0     True
    1    False
    2     None
    dtype: object

    For ``polars``, only columns with ``String`` dtype are modified:

    >>> import pytest
    >>> pl = pytest.importorskip('polars')
    >>> s = pl.Series('s', ['a', 'b', '    '])
    >>> s
    shape: (3,)
    Series: 's' [str]
    [
        "a"
        "b"
        "    "
    ]
    >>> cleaner.fit_transform(s)
    shape: (3,)
    Series: 's' [str]
    [
        "a"
        "b"
        null
    ]
    >>> s = pl.Series('s', ['a', 'b', ''], dtype=pl.Object)
    >>> cleaner.fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Column 's' does not contain strings.
    """

    def fit_transform(self, column, y=None):
        del y
        if not (sbd.is_pandas_object(column) or sbd.is_string(column)):
            raise RejectColumn(f"Column {sbd.name(column)!r} does not contain strings.")
        return self.transform(column)

    def transform(self, column):
        if not (sbd.is_pandas_object(column) or sbd.is_string(column)):
            return column
        column = _trim_whitespace_only(column)
        column = sbd.replace(column, STR_NA_VALUES, sbd.null_value_for(column))
        return column
