import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from . import _dataframe as sbd
from ._dispatch import dispatch
from ._exceptions import RejectColumn


@dispatch
def _with_string_categories(column):
    raise NotImplementedError()


@_with_string_categories.specialize("pandas")
def _with_string_categories_pandas(column):
    categories = column.cat.categories.to_series()
    if pd.api.types.is_string_dtype(
        categories
    ) and not pd.api.types.is_extension_array_dtype(categories):
        return column
    try:
        return column.cat.rename_categories(categories.astype("str"))
    except ValueError:
        # unlikely case that different values in categories have the same
        # string representation: recompute unique categories after casting to
        # string
        is_na = column.isna()
        column = column.astype("str")
        column[is_na] = np.nan
        column = column.astype("category")
        return column


@_with_string_categories.specialize("polars")
def _with_string_categories_polars(column):
    return column


class ToCategorical(BaseEstimator):
    """
    Convert a column to Categorical dtype.

    A string column is converted to a categorical dtype if ``max_categories`` is
    ``None`` or the number of unique values (excluding nulls) in the column is
    smaller or equal to ``max_categories``.

    A categorical input column is also accepted (i.e. it does not result in a
    ``RejectColumn`` exception), regardless of its number of unique values.

    Any other type of column is rejected with a ``RejectColumn`` exception.

    For pandas columns, this transformer also ensures the output's categories have
    the ``'str'`` (object) dtype, not the pandas extension ``StringDtype`` or any
    other dtype. (For polars, categories always have the type ``String``.)

    The output of ``transform`` also always has a Categorical dtype. The categories
    are not necessarily the same across different calls to ``transform``. Indeed,
    scikit-learn estimators do not inspect the dtype's categories but the actual
    values. Converting to a Categorical itself is therefore just a way to mark a
    column and indicate to downstream estimators that this column should be treated
    as categorical. Ensuring they are encoded consistently, handling unseen
    categories at test time, etc. is the responsibility of encoders such as
    ``OneHotEncoder`` and ``LabelEncoder``, or of estimators that handle categories
    themselves such as ``HistGradientBoostingRegressor``.

    Parameters
    ----------
    max_categories : int or None, default=None
        The maximum number of non-null unique values in a string column for it to be
        converted to a Categorical column. String columns with a cardinality
        strictly higher than ``max_categories`` will be rejected with a
        ``RejectColumn`` exception. Note that the categories are not guaranteed to
        be the same across different calls to ``transform``, so subsequent calls to
        ``transform`` may produce outputs with a higher number of categories.

        If ``max_categories`` is ``None``, all string columns are converted.

        If the input column already has a Categorical dtype, it is always accepted
        (and if it is a pandas column its categories are cast to str if necessary),
        regardless of its cardinality.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub._to_categorical import ToCategorical

    A string column with a low cardinality is converted to a categorical column.

    >>> s = pd.Series(['one', 'two', 'three'], name='c')
    >>> to_cat = ToCategorical(max_categories=5)
    >>> to_cat.fit_transform(s)
    0      one
    1      two
    2    three
    Name: c, dtype: category
    Categories (3, object): ['one', 'three', 'two']

    If the input to ``fit`` was accepted, inputs to ``transform`` are always
    converted to Categorical regardless of their cardinality. The dtypes (the list
    of categories) of the outputs may vary.

    >>> s = pd.Series(['four', 'five'], name='c')
    >>> to_cat.transform(s)
    0    four
    1    five
    Name: c, dtype: category
    Categories (2, object): ['five', 'four']

    The number of categories in the output of ``transform`` may be higher than
    ``max_categories``.

    >>> s = pd.Series([f'cat_{i:>02}' for i in range(7)], name='c')
    >>> to_cat.transform(s)
    0    cat_00
    1    cat_01
    2    cat_02
    3    cat_03
    4    cat_04
    5    cat_05
    6    cat_06
    Name: c, dtype: category
    Categories (7, object): ['cat_00', 'cat_01', 'cat_02', 'cat_03', 'cat_04', 'cat_05', 'cat_06']

    A Categorical input column is always accepted regardless of its cardinality.

    >>> to_cat = ToCategorical(max_categories=5)
    >>> s = pd.Series([f'cat_{i:>02}' for i in range(7)], name='c').astype('category')
    >>> to_cat.fit_transform(s)
    0    cat_00
    1    cat_01
    2    cat_02
    3    cat_03
    4    cat_04
    5    cat_05
    6    cat_06
    Name: c, dtype: category
    Categories (7, object): ['cat_00', 'cat_01', 'cat_02', 'cat_03', 'cat_04', 'cat_05', 'cat_06']

    A string column with a high cardinality or a column that does not contain
    strings nor categories will be rejected.

    >>> to_cat.fit_transform(s.astype('str'))
    Traceback (most recent call last):
        ...
    skrub._exceptions.RejectColumn: Cardinality of column 'c' is > max_categories, which is: 5.

    Null values do not count toward the cardinality of the input column.

    >>> s = pd.Series(
    ...     [f"cat_{i:>02}" for i in range(to_cat.max_categories)]
    ...     + [None, None, None],
    ...     name="c",
    ... )
    >>> to_cat.fit_transform(s)
    0    cat_00
    1    cat_01
    2    cat_02
    3    cat_03
    4    cat_04
    5       NaN
    6       NaN
    7       NaN
    Name: c, dtype: category
    Categories (5, object): ['cat_00', 'cat_01', 'cat_02', 'cat_03', 'cat_04']

    Columns that are neither strings nor categories are rejected.

    >>> to_cat.fit_transform(pd.Series([1.1, 2.2], name='c'))
    Traceback (most recent call last):
        ...
    skrub._exceptions.RejectColumn: Column 'c' does not contain strings or categories.

    This also applies to ``object`` columns that do not contain only strings.

    >>> s = pd.Series(['one', 1], name='c')
    >>> to_cat.fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._exceptions.RejectColumn: Column 'c' does not contain strings or categories.

    The categories are always converted to strings and the pandas ``object`` dtype.
    This is to avoid errors in scikit-learn encoders or estimators which may occur
    in the presence of ``pd.NA``, or when a categorical column has categories that
    mix strings and other types.

    >>> s = pd.Series(['cat A', 'cat B', None], name='c', dtype='string')
    >>> s
    0    cat A
    1    cat B
    2     <NA>
    Name: c, dtype: string
    >>> s.astype('category')
    0    cat A
    1    cat B
    2     <NA>
    Name: c, dtype: category
    Categories (2, string): [cat A, cat B]
    >>> _.cat.categories.dtype
    string[python]
    >>> to_cat.fit_transform(s)
    0    cat A
    1    cat B
    2      NaN
    Name: c, dtype: category
    Categories (2, object): ['cat A', 'cat B']
    >>> _.cat.categories.dtype
    dtype('O')

    Unlike the output of ``pandas.Series.astype()`` which preserves the use of the
    extension dtype, here the categories in the output of ``fit_transform`` has
    dtype ``object`` and missing values in the column are encoded with ``np.nan``,
    not ``pd.NA``.

    This conversion is also applied when the input column is already categorical.

    >>> s = pd.Series(['cat A', 'cat B', None], name='c', dtype='string').astype('category')
    >>> s
    0    cat A
    1    cat B
    2     <NA>
    Name: c, dtype: category
    Categories (2, string): [cat A, cat B]
    >>> to_cat.fit_transform(s)
    0    cat A
    1    cat B
    2      NaN
    Name: c, dtype: category
    Categories (2, object): ['cat A', 'cat B']

    Similarly, non-string categories are converted to strings.

    >>> s = pd.Series([1, 2], name='c').astype('category')
    >>> s
    0    1
    1    2
    Name: c, dtype: category
    Categories (2, int64): [1, 2]
    >>> to_cat.fit_transform(s)
    0    1
    1    2
    Name: c, dtype: category
    Categories (2, object): ['1', '2']

    Above we can see that the output categories are strings.

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
    >>> to_cat.fit_transform(s)
    0    C()
    1    C()
    Name: c, dtype: category
    Categories (1, object): ['C()']

    Most of those considerations do not apply to polars; in polars all Categorical
    columns have Strings as the type of their categories. Only String columns are
    considered for conversion.

    >>> import pytest
    >>> pl = pytest.importorskip('polars')
    >>> s = pl.Series('c', ['one', 'two', None], dtype=pl.String)
    >>> to_cat.fit_transform(s)
    shape: (3,)
    Series: 'c' [cat]
    [
        "one"
        "two"
        null
    ]
    >>> s = pl.Series('c', ['one', 'two', None], dtype=pl.Categorical)
    >>> to_cat.fit_transform(s)
    shape: (3,)
    Series: 'c' [cat]
    [
        "one"
        "two"
        null
    ]
    >>> s = pl.Series('c', ['one', 'two', None], dtype=pl.Object)
    >>> to_cat.fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._exceptions.RejectColumn: Column 'c' does not contain strings or categories.

    When converting strings, the output of ``ToCategorical`` is always
    ``Categorical``, but an ``Enum`` column is also accepted and passed through
    unchanged.

    >>> s = pl.Series('c', ['one', 'two', None], dtype=pl.Enum(['one', 'two', 'three']))
    >>> to_cat.fit_transform(s)
    shape: (3,)
    Series: 'c' [enum]
    [
        "one"
        "two"
        null
    ]
    """

    __single_column_transformer__ = True

    def __init__(self, max_categories=None):
        self.max_categories = max_categories

    def fit_transform(self, column):
        if sbd.is_categorical(column):
            return _with_string_categories(column)
        if not sbd.is_string(column):
            raise RejectColumn(
                f"Column {sbd.name(column)!r} does not contain strings or categories."
            )
        n_categories = len(sbd.drop_nulls(sbd.unique(column)))
        if self.max_categories is not None and self.max_categories < n_categories:
            raise RejectColumn(
                f"Cardinality of column {sbd.name(column)!r} "
                f"is > max_categories, which is: {self.max_categories}."
            )
        return _with_string_categories(sbd.to_categorical(column))

    def transform(self, column):
        return _with_string_categories(sbd.to_categorical(column))

    def fit(self, column):
        self.fit_transform(column)
        return self
