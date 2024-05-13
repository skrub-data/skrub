import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from . import _dataframe as sbd
from ._exceptions import RejectColumn


class PandasStringDtypeToObject(BaseEstimator):
    """
    Convert ``StringDtype`` columns to ``object`` columns.

    Scikit-learn estimators and encoders do not yet handle pandas extension
    dtypes, especially in the presence of missing values (represented by
    ``pd.NA``). This transformer converts a column with the extension dtype
    ``StringDtype`` (missing values represented by ``pd.NA``) with a column
    with dtype ``object`` (missing values represented by ``np.nan``). It also
    replaces any missing values in an ``object`` column with ``np.nan``, to
    make sure the output does not contain ``pd.NA``.

    Polars ``String`` columns are passed through unchanged.

    Non-string columns are rejected with a ``RejectColumn`` exception.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub._pandas_string_dtype_to_object import PandasStringDtypeToObject
    >>> s_to_obj = PandasStringDtypeToObject()

    A column with the dtype ``StringDtype`` is converted to ``object``:

    >>> s = pd.Series(['one', 'two', None], name='s', dtype='string')
    >>> s
    0     one
    1     two
    2    <NA>
    Name: s, dtype: string
    >>> s_to_obj.fit_transform(s)
    0    one
    1    two
    2    NaN
    Name: s, dtype: object

    In an ``object`` column, ``pd.NA`` is also replaced with ``np.nan``:

    >>> s = pd.Series(['one', 'two', pd.NA], name='s', dtype='str')
    >>> s
    0     one
    1     two
    2    <NA>
    Name: s, dtype: object
    >>> s_to_obj.fit_transform(s)
    0    one
    1    two
    2    NaN
    Name: s, dtype: object

    Non-string columns are rejected.

    >>> s = pd.Series(['one', 2.0, None], name='s')
    >>> s
    0     one
    1     2.0
    2    None
    Name: s, dtype: object
    >>> s_to_obj.fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._exceptions.RejectColumn: Column 's' does not have a str or string dtype.

    Polars columns are accepted but never modified

    >>> import pytest
    >>> pl = pytest.importorskip('polars')
    >>> s = pl.Series('s', ['one', 'two', None])
    >>> s
    shape: (3,)
    Series: 's' [str]
    [
        "one"
        "two"
        null
    ]
    >>> s_to_obj.fit_transform(s) is s
    True
    """

    __single_column_transformer__ = True

    def fit_transform(self, column):
        if not sbd.is_string(column):
            raise RejectColumn(
                f"Column {sbd.name(column)!r} does not have a str or string dtype."
            )
        return self.transform(column)

    def transform(self, column):
        if not sbd.is_pandas(column):
            return column
        is_na = column.isna()
        if column.dtype == pd.StringDtype():
            column = column.astype("str")
            column[is_na] = np.nan
            return column
        if not is_na.any():
            return column
        return column.fillna(np.nan)
