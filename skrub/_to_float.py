import re

from . import _dataframe as sbd
from ._dispatch import dispatch, raise_dispatch_unregistered_type
from ._single_column_transformer import RejectColumn, SingleColumnTransformer

__all__ = ["ToFloat"]


def _build_number_regex(decimal, thousand):
    d = re.escape(decimal)
    t = re.escape(thousand)

    integer = rf"(?:\d+|\d{{1,3}}(?:{t}\d{{3}})+)"
    decimal_part = rf"(?:{d}\d+)?"
    scientific = r"(?:[eE][+-]?\d+)?"

    return rf"""
        ^
        \(?
        [+-]?
        {integer}
        {decimal_part}
        {scientific}
        \)?
        $
    """


@dispatch
def _str_is_valid_number(col, number_re):
    raise_dispatch_unregistered_type(col, kind="Series")


@_str_is_valid_number.specialize("pandas", argument_type="Column")
def _str_is_valid_number_pandas(col, number_re):
    if not col.str.match(number_re, na=False).all():
        raise RejectColumn(f"The pattern could not match the column {sbd.name(col)!r}.")
    return True


@_str_is_valid_number.specialize("polars", argument_type="Column")
def _str_is_valid_number_polars(col, number_re):
    if not col.str.contains(number_re.pattern).all():
        raise RejectColumn(f"The pattern could not match the column {sbd.name(col)!r}.")
    return True


@dispatch
def _str_replace(col, strict=True):
    raise_dispatch_unregistered_type(col, kind="Series")


@_str_replace.specialize("pandas", argument_type="Column")
def _str_replace_pandas(col, decimal, thousand):
    col = col.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    col = col.str.replace(thousand, "", regex=False)
    return col.str.replace(decimal, ".", regex=False)


@_str_replace.specialize("polars", argument_type="Column")
def _str_replace_polars(col, decimal, thousand):
    col = col.str.replace_all(r"^\((.*)\)$", r"-$1")
    col = col.str.replace_all(thousand, "")
    return col.str.replace_all(f"[{decimal}]", ".")


class ToFloat(SingleColumnTransformer):
    """
    Convert a column to 32-bit floating-point numbers.

    No conversion is attempted if the column has a datetime or categorical
    dtype; a ``RejectColumn`` exception is raised.

    Otherwise, we attempt to convert the column to float32. If the conversion
    fails the column is rejected (a ``RejectColumn`` exception is raised).

    For pandas, the output is always ``np.float32``, not the extension dtype
    ``pd.Float64Dtype``. We do this conversion because most scikit-learn
    estimators cannot handle those dtypes correctly yet, especially in the
    presence of missing values (represented by ``pd.NA`` in such columns).

    During ``transform``, entries for which conversion fails are replaced by
    null values.

    Parameters
    ----------
    decimal : str, default='.'
        Character to recognize as the decimal separator when converting from
        strings to floats. Other possible decimal separators are removed from
        the strings before conversion.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub._to_float import ToFloat

    A column that does not contain floats is converted if possible:

    >>> s = pd.Series(['1.1', None, '3.3'], name='x')
    >>> s
    0     1.1
    1    ...
    2     3.3
    Name: x, dtype: ...
    >>> s[0]
    '1.1'
    >>> to_float = ToFloat()
    >>> float_s = to_float.fit_transform(s)
    >>> float_s
    0    1.1
    1    NaN
    2    3.3
    Name: x, dtype: float32
    >>> float_s[0]            # doctest: +SKIP
    np.float32(1.1)

    Note that a column such as the example above may easily occur as the output
    of ``CleanNullStrings``.

    A numeric column will also be converted to floats:

    >>> s = pd.Series([1, 2, 3])
    >>> s
    0    1
    1    2
    2    3
    dtype: int64
    >>> to_float.fit_transform(s)
    0    1.0
    1    2.0
    2    3.0
    dtype: float32

    Boolean columns are treated as numbers:

    >>> s = pd.Series([True, False], name='b')
    >>> s
    0     True
    1    False
    Name: b, dtype: bool
    >>> to_float.fit_transform(s)
    0    1.0
    1    0.0
    Name: b, dtype: float32

    >>> s = pd.Series([True, None], name='b', dtype='boolean')
    >>> s
    0    True
    1    <NA>
    Name: b, dtype: boolean
    >>> to_float.fit_transform(s)
    0    1.0
    1    NaN
    Name: b, dtype: float32
    >>> s = pd.Series([True, None], name='b')
    >>> s
    0    True
    1    None
    Name: b, dtype: object
    >>> to_float.fit_transform(s)
    0    1.0
    1    NaN
    Name: b, dtype: float32

    float64 columns are converted to float32:

    >>> s = pd.Series([1.1, 2.2])
    >>> s
    0    1.1
    1    2.2
    dtype: float64
    >>> to_float.fit_transform(s)
    0    1.1
    1    2.2
    dtype: float32

    Float64Dtype and Float32Dtype are cast to ``np.float32``. We do this because
    most scikit-learn estimators cannot handle ``pd.Float32Dtype`` correctly
    yet, especially in the presence of missing values (represented by ``pd.NA``
    in such columns).

    >>> s = pd.Series([1.1, 2.2, None], dtype='Float32')
    >>> s
    0     1.1
    1     2.2
    2    <NA>
    dtype: Float32
    >>> to_float.fit_transform(s)
    0    1.1
    1    2.2
    2    NaN
    dtype: float32

    Notice that ``pd.NA`` has been replaced by ``np.nan``.

    Columns that cannot be cast to numbers are rejected:

    >>> s = pd.Series(['1.1', '2.2', 'hello'], name='x')
    >>> to_float.fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._single_column_transformer.RejectColumn: Could not convert column 'x' to numbers.

    Once a column has been accepted, all calls to ``transform`` will result in the
    same output dtype. Values that fail to be converted become null values.

    >>> to_float = ToFloat().fit(pd.Series([1, 2]))
    >>> to_float.transform(pd.Series(['3.3', 'hello']))
    0    3.3
    1    NaN
    dtype: float32

    Categorical and datetime columns are always rejected:

    >>> s = pd.Series(['1.1', '2.2'], dtype='category', name='s')
    >>> s
    0    1.1
    1    2.2
    Name: s, dtype: category
    Categories (2, ...): ['1.1', '2.2']
    >>> to_float.fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._single_column_transformer.RejectColumn: Refusing to cast column 's' with dtype 'category' to numbers.
    >>> to_float.fit_transform(pd.to_datetime(pd.Series(['2024-05-13'], name='s')))
    Traceback (most recent call last):
        ...
    skrub._single_column_transformer.RejectColumn: Refusing to cast column 's' with dtype 'datetime64[...]' to numbers.

    float32 columns are passed through:

    >>> s = pd.Series([1.1, None], dtype='float32')
    >>> to_float.fit_transform(s) is s
    True

    Handling parentheses around negative numbers
    >>> s = pd.Series(["-1,234.56", "1,234.56", "(1,234.56)"], name='parens')
    >>> to_float.fit_transform(s)
    0   -1234.5...
    1    1234.5...
    2   -1234.5...
    dtype: float32

    Scientific notation
    >>> s = pd.Series(["1.23e+4", "1.23E+4"], name="x")
    >>> ToFloat(decimal=".").fit_transform(s)
    0    12300.0
    1    12300.0
    Name: x, dtype: float32

    Space or apostrophe as thousand separator
    >>> s = pd.Series(["4 567,89", "4'567,89"], name="x")
    >>> ToFloat(decimal=",").fit_transform(s)
    0    4567.8...
    1    4567.8...
    Name: x, dtype: float32
    """  # noqa: E501

    def __init__(self, decimal=".", thousand=None):
        super().__init__()
        self.decimal = decimal
        self.thousand = thousand if thousand is not None else ""

    def fit_transform(self, column, y=None):
        """Fit the encoder and transform a column.

        Parameters
        ----------
        column : pandas or polars Series
            The input to transform.

        y : None
            Ignored.

        Returns
        -------
        transformed : pandas or polars Series
            The input transformed to Float32.
        """
        del y
        self.all_outputs_ = [sbd.name(column)]
        if self.thousand == self.decimal:
            raise ValueError("The thousand and decimal separators must differ.")

        if sbd.is_any_date(column) or sbd.is_categorical(column):
            raise RejectColumn(
                f"Refusing to cast column {sbd.name(column)!r} "
                f"with dtype '{sbd.dtype(column)}' to numbers."
            )
        try:
            if sbd.is_string(column):
                self._number_re_ = re.compile(
                    _build_number_regex(self.decimal, self.thousand),
                    re.VERBOSE,
                )
                _str_is_valid_number(column, self._number_re_)
                column = _str_replace(
                    column, decimal=self.decimal, thousand=self.thousand
                )
            numeric = sbd.to_float32(column, strict=True)
            return numeric
        except Exception as e:
            raise RejectColumn(
                f"Could not convert column {sbd.name(column)!r} to numbers."
            ) from e

    def transform(self, column):
        """Transform a column.

        Parameters
        ----------
        column : pandas or polars Series
            The input to transform.

        Returns
        -------
        transformed : pandas or polars Series
            The input transformed to Float32.
        """
        return sbd.to_float32(column, strict=False)
