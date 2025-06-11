import numpy as np

from . import _dataframe as sbd
from ._dispatch import dispatch
from ._on_each_column import RejectColumn, SingleColumnTransformer

__all__ = ["CleanCategories"]


@dispatch
def _with_string_categories(col):
    raise NotImplementedError()


@_with_string_categories.specialize("pandas", argument_type="Column")
def _with_string_categories_pandas(col):
    categories = col.cat.categories.to_series()
    if sbd.is_string(categories) and not sbd.is_pandas_extension_dtype(categories):
        return col
    try:
        return col.cat.rename_categories(categories.astype("str"))
    except ValueError:
        # unlikely case that different values in categories have the same
        # string representation: recompute unique categories after casting to
        # string
        is_na = col.isna()
        col = col.astype("str")
        col[is_na] = np.nan
        col = col.astype("category")
        return col


@_with_string_categories.specialize("polars", argument_type="Column")
def _with_string_categories_polars(col):
    return col


class CleanCategories(SingleColumnTransformer):
    """
    Preprocess a categorical column.

    For pandas columns, this transformer ensures that the output's
    ``.cat.categories`` attribute contains only strings and has the ``object``
    dtype.

    Pandas allows anything as categories but scikit-learn encoders raise an
    exception when categories mix strings and numbers or use any other type, so
    we make sure they are strings. Note that this can result in collapsing 2
    categories in the edge case that they are different but have the same string
    representation.

    Pandas categorical columns that store their categories with the
    ``StringDtype`` extension dtype represent missing values with ``pd.NA``,
    which scikit-learn encoders and estimators cannot handle, so we make sure
    categories are stored with the ``object`` dtype.

    Polars categories are always ``String`` and they require no special
    preprocessing. They are accepted (do not result a ``RejectColumn``
    exception) by this transformer but are passed through unchanged.

    Any column that does not have a categorical dtype is rejected with a
    ``RejectColumn`` exception (see ``ToCategorical`` for converting columns to
    a categorical dtype).

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub._clean_categories import CleanCategories

    A categorical column that already stores its categories as strings in an
    index with dtype ``object``, as expected by scikit-learn encoders, is
    passed through unchanged:

    >>> s = pd.Series(['one', 'two', 'three'], name='c', dtype='category')
    >>> s
    0      one
    1      two
    2    three
    Name: c, dtype: category
    Categories (3, object): ['one', 'three', 'two']
    >>> cleaner = CleanCategories()
    >>> cleaner.fit_transform(s)
    0      one
    1      two
    2    three
    Name: c, dtype: category
    Categories (3, object): ['one', 'three', 'two']
    >>> cleaner.fit_transform(s) is s
    True

    Categories stored with the ``StringDtype`` dtype are converted to ``object``:

    >>> s = pd.Series(['cat A', 'cat B', None], name='c', dtype='string')
    >>> s = s.astype('category')
    >>> s
    0    cat A
    1    cat B
    2     <NA>
    Name: c, dtype: category
    Categories (2, string): [cat A, cat B]
    >>> _.cat.categories.dtype
    string[python]
    >>> cleaner.fit_transform(s)
    0    cat A
    1    cat B
    2      NaN
    Name: c, dtype: category
    Categories (2, object): ['cat A', 'cat B']
    >>> _.cat.categories.dtype
    dtype('O')

    Non-string categories are converted to strings:

    >>> s = pd.Series([1, 2], name='c').astype('category')
    >>> s
    0    1
    1    2
    Name: c, dtype: category
    Categories (2, int64): [1, 2]
    >>> cleaner.fit_transform(s)
    0    1
    1    2
    Name: c, dtype: category
    Categories (2, object): ['1', '2']

    We can see above that the output categories are strings.

    Note: this can result in some categories being collapsed in the edge case where
    different categories have the same string representation, as shown below.

    >>> class C:
    ...     def __repr__(self):
    ...         return 'C()'

    >>> s = pd.Series([C(), C()], name='c').astype('category')
    >>> s
    0    C()
    1    C()
    Name: c, dtype: category
    Categories (2, object): [C(), C()]
    >>> cleaner.fit_transform(s)
    0    C()
    1    C()
    Name: c, dtype: category
    Categories (1, object): ['C()']

    A non-categorical column is rejected:

    >>> s = pd.Series(['a', 'b', 'c'], name='c')
    >>> cleaner.fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Column 'c' is not categorical.

    However once a column has been accepted, the output of ``transform`` will
    always have a categorical dtype:

    >>> cleaner.fit(s.astype('category'))
    CleanCategories()
    >>> s
    0    a
    1    b
    2    c
    Name: c, dtype: object
    >>> cleaner.transform(s)
    0    a
    1    b
    2    c
    Name: c, dtype: category
    Categories (3, object): ['a', 'b', 'c']

    The categories themselves are not checked and may vary depending on the
    input; handling unseen categories is the responsibility of encoders such as
    ``OneHotEncoder`` or ``LabelEncoder``.

    >>> cleaner.transform(pd.Series(['x', 'y', None], name='c'))
    0      x
    1      y
    2    NaN
    Name: c, dtype: category
    Categories (2, object): ['x', 'y']

    The category dtype transformations do not apply to polars; in polars all
    Categorical columns have Strings as the type of their categories.

    >>> import pytest
    >>> pl = pytest.importorskip('polars')
    >>> s = pl.Series('c', ['one', 'two', None], dtype=pl.Categorical)
    >>> cleaner.fit_transform(s) is s
    True
    >>> s = pl.Series('c', ['one', 'two', None], dtype=pl.Enum(['one', 'two', 'three']))
    >>> cleaner.fit_transform(s) is s
    True
    """

    def fit_transform(self, column, y=None):
        del y
        if not sbd.is_categorical(column):
            raise RejectColumn(f"Column {sbd.name(column)!r} is not categorical.")
        return _with_string_categories(column)

    def transform(self, column):
        return _with_string_categories(sbd.to_categorical(column))
