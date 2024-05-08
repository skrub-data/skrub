import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from . import _dataframe as sbd
from ._dispatch import dispatch
from ._exceptions import RejectColumn


@dispatch
def _choose_float_dtype_for(col):
    raise NotImplementedError()


@_choose_float_dtype_for.specialize("pandas")
def _choose_float_dtype_for_pandas(col):
    if not pd.api.types.is_float_dtype(col):
        return np.float32
    if not pd.api.types.is_extension_array_dtype(col):
        return col.dtype
    if col.dtype == pd.Float64Dtype():
        return np.float64
    return np.float32


@_choose_float_dtype_for.specialize("polars")
def _choose_float_dtype_for_polars(col):
    import polars as pl

    if col.dtype.is_float():
        return col.dtype
    return pl.Float32


@dispatch
def _cast_to_float_dtype(col, dtype, strict):
    raise NotImplementedError()


@_cast_to_float_dtype.specialize("pandas")
def _cast_to_float_dtype_pandas(col, dtype, strict):
    if not pd.api.types.is_numeric_dtype(col):
        col = pd.to_numeric(col, errors="raise" if strict else "coerce")
    if col.dtype == dtype:
        return col
    return col.astype(dtype)


@_cast_to_float_dtype.specialize("polars")
def _cast_to_float_dtype_polars(col, dtype, strict):
    return col.cast(dtype, strict=strict)


def _to_float(col, dtype, strict):
    if dtype is None:
        dtype = _choose_float_dtype_for(col)
    if sbd.dtype(col) == dtype:
        return col
    return _cast_to_float_dtype(col, dtype=dtype, strict=strict)


class ToFloat(BaseEstimator):
    """
    Convert a column to floating-point numbers.

    - No conversion is attempted if the column has a datetime or categorical dtype.
    - If the column does not have a floating-point dtype (e.g. its dtype is
      string, int, bool, …), we attempt to convert it to float32. If the conversion
      fails the column is rejected (a ``RejectColumn`` exception is raised).
    - pandas columns of the ``Float32Dtype`` or ``Float64Dtype`` extension
      dtypes are converted to the numpy dtype of the same bit width
      (``Float32Dtype`` → ``np.float32``, ``Float64Dtype`` → ``np.float64``).
      We do this because most scikit-learn estimators cannot handle those
      dtypes correctly yet, especially in the presence of missing values
      (represented by ``pd.NA`` in such columns).
    - other floating-point columns are passed through.

    During ``transform`` the output dtype is always the same (given by
    ``self.output_dtype_``) and entries for which conversion fails are replaced by
    null values.

    Attributes
    ----------
    output_dtype_ : data type.
        The dtype of the output of ``transform``, such as ``numpy.float32`` or
        ``polars.Float32``.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub._to_float import ToFloat

    >>> s = pd.Series(['1.1', None, '3.3'], name='x')
    >>> s
    0     1.1
    1    None
    2     3.3
    Name: x, dtype: object
    >>> s[0]
    '1.1'
    >>> to_float = ToFloat()
    >>> to_float.fit_transform(s)
    0    1.1
    1    NaN
    2    3.3
    Name: x, dtype: float32
    >>> _[0]
    1.1

    Note that a column such as the example above may easily occur as the output
    of ``CleanNullStrings``.

    The output dtype is recorded:

    >>> to_float.output_dtype_
    dtype('float32')

    Columns that are already floats are left unchanged, except for pandas extension
    dtypes.

    >>> s = pd.Series([1.1, None, 3.3], name='x')
    >>> s
    0    1.1
    1    NaN
    2    3.3
    Name: x, dtype: float64
    >>> to_float.fit_transform(s) is s
    True
    >>> to_float.output_dtype_
    dtype('float64')

    Float64Dtype and Float32Dtype are cast to a numpy float dtype of the same bit
    width. We do this because most scikit-learn estimators cannot handle those
    dtypes correctly yet, especially in the presence of missing values (represented
    by ``pd.NA`` in such columns).

    >>> s = pd.Series([1.1, 2.2, None], dtype="Float64")
    >>> s
    0     1.1
    1     2.2
    2    <NA>
    dtype: Float64
    >>> to_float.fit_transform(s)
    0    1.1
    1    2.2
    2    NaN
    dtype: float64

    Notice that ``pd.NA`` has been replaced by ``np.nan`` and the bit width has been
    preserved.

    Columns that cannot be cast to numbers are rejected:

    >>> s = pd.Series(['1.1', '2.2', 'hello'], name='x')
    >>> to_float.fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._exceptions.RejectColumn: Could not convert column 'x' to numbers.

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

    Once a column has been accepted, all call to ``transform`` will result in the
    same output type. Values that fail to be converted become null values.

    >>> to_float = ToFloat().fit(pd.Series([1, 2]))
    >>> to_float.transform(pd.Series(['3.3', 'hello']))
    0    3.3
    1    NaN
    dtype: float32
    """

    __single_column_transformer__ = True

    def fit_transform(self, column):
        if sbd.is_any_date(column) or sbd.is_categorical(column):
            raise RejectColumn(
                f"Refusing to cast column {sbd.name(column)!r} "
                f"with dtype {sbd.dtype(column)} to numbers."
            )
        try:
            numeric = _to_float(column, dtype=None, strict=True)
            self.output_dtype_ = sbd.dtype(numeric)
            return numeric
        except Exception as e:
            raise RejectColumn(
                f"Could not convert column {sbd.name(column)!r} to numbers."
            ) from e

    def transform(self, column):
        return _to_float(column, dtype=self.output_dtype_, strict=False)

    def fit(self, column):
        self.fit_transform(column)
        return self
