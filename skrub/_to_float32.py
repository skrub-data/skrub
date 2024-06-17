from . import _dataframe as sbd
from ._on_each_column import RejectColumn, SingleColumnTransformer

__all__ = ["ToFloat32"]


class ToFloat32(SingleColumnTransformer):
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

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub._to_float32 import ToFloat32

    A column that does not contain floats is converted if possible:

    >>> s = pd.Series(['1.1', None, '3.3'], name='x')
    >>> s
    0     1.1
    1    None
    2     3.3
    Name: x, dtype: object
    >>> s[0]
    '1.1'
    >>> to_float = ToFloat32()
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
    skrub._on_each_column.RejectColumn: Could not convert column 'x' to numbers.

    Once a column has been accepted, all calls to ``transform`` will result in the
    same output dtype. Values that fail to be converted become null values.

    >>> to_float = ToFloat32().fit(pd.Series([1, 2]))
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
    Categories (2, object): ['1.1', '2.2']
    >>> to_float.fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Refusing to cast column 's' with dtype 'category' to numbers.
    >>> to_float.fit_transform(pd.to_datetime(pd.Series(['2024-05-13'], name='s')))
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Refusing to cast column 's' with dtype 'datetime64[ns]' to numbers.

    float32 columns are passed through:

    >>> s = pd.Series([1.1, None], dtype='float32')
    >>> to_float.fit_transform(s) is s
    True
    """  # noqa: E501

    def fit_transform(self, column, y=None):
        del y
        if sbd.is_any_date(column) or sbd.is_categorical(column):
            raise RejectColumn(
                f"Refusing to cast column {sbd.name(column)!r} "
                f"with dtype '{sbd.dtype(column)}' to numbers."
            )
        try:
            numeric = sbd.to_float32(column, strict=True)
            return numeric
        except Exception as e:
            raise RejectColumn(
                f"Could not convert column {sbd.name(column)!r} to numbers."
            ) from e

    def transform(self, column):
        return sbd.to_float32(column, strict=False)
