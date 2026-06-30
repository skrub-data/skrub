import fnmatch
import numbers
import re

from .. import _dataframe as sbd
from ._base import Filter, NameFilter

__all__ = [
    "glob",
    "regex",
    "numeric",
    "integer",
    "float",
    "has_dtype",
    "any_date",
    "categorical",
    "string",
    "object",
    "boolean",
    "cardinality_below",
    "has_nulls",
]

#
# Selectors based on column names
#


def glob(pattern):
    """Select columns by name with Unix shell style 'glob' pattern.

    **When to use:**
    Use this selector for simple wildcard patterns on column names.
    This is useful for selecting columns with predictable naming patterns
    (e.g., all columns ending in '_mm', or starting with 'feature_').
    For more complex patterns, consider :func:`regex`.

    Pattern matching is case-sensitive and interpreted as described in
    ``fnmatch.fnmatchcase``::

        *       matches everything
        ?       matches any single character
        [seq]   matches any character in seq
        [!seq]  matches any char not in seq

    Parameters
    ----------
    pattern : str
        A glob pattern to match column names.

    See Also
    --------
    regex :
        Select columns by name using regular expressions.
        Use this for complex patterns that glob cannot express.
    filter_names :
        Select columns based on custom name-based criteria.

    Examples
    --------
    >>> from skrub import selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "height_mm": [297.0, 420.0],
    ...         "width_mm": [210.0, 297.0],
    ...         "kind": ["A4", "A3"],
    ...         "ID": [4, 3],
    ...     }
    ... )

    Select all columns with a simple wildcard:

    >>> s.select(df, s.glob('*'))
       height_mm  width_mm kind  ID
    0      297.0     210.0   A4   4
    1      420.0     297.0   A3   3

    Select columns matching a pattern:

    >>> s.select(df, s.glob('*_mm'))
       height_mm  width_mm
    0      297.0     210.0
    1      420.0     297.0

    Use character classes to match specific patterns:

    >>> s.select(df, s.glob('[a-z]*_mm'))
       height_mm  width_mm
    0      297.0     210.0
    1      420.0     297.0

    Combine with other selectors:

    >>> s.select(df, s.glob('*_mm') | s.glob('ID'))
       height_mm  width_mm  ID
    0      297.0     210.0   4
    1      420.0     297.0   3

    """
    return NameFilter(fnmatch.fnmatchcase, args=(pattern,), name="glob")


def _regex(col_name, pattern, flags=0):
    return re.match(pattern, col_name, flags=flags) is not None


def regex(pattern, flags=0):
    """Select columns by name with a regular expression.

    **When to use:**
    Use this selector for complex name patterns that glob patterns cannot express.
    This is useful for selecting columns with specific naming conventions or
    patterns that require regular expression features (e.g., columns matching
    '^feature_[0-9]+$'). For simple wildcard patterns, prefer :func:`glob`.

    Parameters
    ----------
    pattern : str or compiled regex
        A regular expression pattern to match column names. Can be a string pattern
        or a compiled regular expression object.
    flags : int, optional
        Regular expression flags as described in the ``re`` module documentation:
        https://docs.python.org/3/library/re.html#flags

    See Also
    --------
    glob :
        Select columns by name with Unix shell-style wildcard patterns.
        Use this for simpler patterns.
    filter_names :
        Select columns based on custom name-based criteria.

    Notes
    -----
    A column is selected if ``re.match(col_name, pattern, flags)`` returns a match.
    Note that ``re.match`` only matches at the beginning of the string. Use '$' at
    the end to require matching until the end of the column name.

    Examples
    --------
    >>> from skrub import selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "height_mm": [297.0, 420.0],
    ...         "width_mm": [210.0, 297.0],
    ...         "kind": ["A4", "A3"],
    ...         "ID": [4, 3],
    ...     }
    ... )

    Select columns matching a pattern:

    >>> s.select(df, s.regex('.*_mm'))
       height_mm  width_mm
    0      297.0     210.0
    1      420.0     297.0

    Match at the beginning of the column name (no need for ^ prefix):

    >>> s.select(df, s.regex('wid'))
       width_mm
    0     210.0
    1     297.0

    Use '$' to require matching until the end of the column name:

    >>> s.select(df, s.regex('wid$'))
    Empty DataFrame
    Columns: []
    Index: [0, 1]

    Use regex flags for case-insensitive matching:

    >>> import re
    >>> s.select(df, s.regex('id', flags=re.I))
       ID
    0   4
    1   3

    Flags can also be embedded in the pattern:

    >>> s.select(df, s.regex('(?i)id'))
       ID
    0   4
    1   3

    Or use a compiled pattern:

    >>> s.select(df, s.regex(re.compile('id', re.I)))
       ID
    0   4
    1   3

    Combine with other selectors:

    >>> s.select(df, s.regex('^[a-z]+_mm$') | s.glob('ID'))
       height_mm  width_mm  ID
    0      297.0     210.0   4
    1      420.0     297.0   3

    """
    kwargs = {"flags": flags} if flags != 0 else {}
    return NameFilter(_regex, args=(pattern,), kwargs=kwargs, name="regex")


#
# Selectors based on data types
#


def numeric():
    """
    Select columns that have a numeric data type.

    **When to use:**
    Use this selector to find all numeric columns for scaling, normalization,
    or statistical analysis. This includes both integer and floating-point types
    but excludes Boolean columns (which often need different handling).

    This selector matches both integer and floating-point columns, equivalent to
    ``integer() | float()``.

    Notes
    -----
    Boolean columns are intentionally excluded because they typically require
    different preprocessing strategies than numeric features (e.g., encoding
    rather than scaling).

    See Also
    --------
    integer :
        Select integer columns only.
    float :
        Select floating-point columns only.
    boolean :
        Select Boolean columns.

    Examples
    --------
    >>> from skrub import selectors as s
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame(
    ...     dict(
    ...         f64=[1.1],
    ...         F64=pd.Series([2.3]).convert_dtypes(),
    ...         i64=[2],
    ...         I64=pd.Series([2]).convert_dtypes(),
    ...         i8=np.int8(3),
    ...         bool_=[True],
    ...         Bool_=pd.Series([True]).convert_dtypes(),
    ...         str_=["hello"],
    ...     )
    ... )

    >>> df
       f64  F64  i64  I64  i8  bool_  Bool_   str_
    0  1.1  2.3    2    2   3   True   True  hello

    Select all numeric columns (note: booleans are excluded):

    >>> s.select(df, s.numeric())
       f64  F64  i64  I64  i8
    0  1.1  2.3    2    2   3

    Combine with :func:`boolean` to include Boolean columns:

    >>> s.select(df, s.numeric() | s.boolean())
       f64  F64  i64  I64  i8  bool_  Bool_
    0  1.1  2.3    2    2   3   True   True

    """
    return Filter(sbd.is_numeric, name="numeric")


def integer():
    """
    Select columns that have an integer data type.

    **When to use:**
    Use this selector when you specifically need integer-typed columns,
    excluding floating-point and Boolean types. This is useful for selecting
    discrete numeric features or ID-like columns.

    This selector selects only integer columns but not Boolean columns.
    Note that ``integer() | float()`` is equivalent to :func:`numeric()`.

    Notes
    -----
    Boolean columns are intentionally excluded because they typically require
    different preprocessing strategies than numeric features.

    See Also
    --------
    numeric :
        Select all numeric columns (integer and float).
        Use this to select both integer and floating-point columns together.
    float :
        Select floating-point columns only.
    boolean :
        Select Boolean columns.

    Examples
    --------
    >>> from skrub import selectors as s
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame(
    ...     dict(
    ...         f64=[1.1],
    ...         F64=pd.Series([2.3]).convert_dtypes(),
    ...         i64=[2],
    ...         I64=pd.Series([2]).convert_dtypes(),
    ...         i8=np.int8(3),
    ...         bool_=[True],
    ...         Bool_=pd.Series([True]).convert_dtypes(),
    ...         str_=["hello"],
    ...     )
    ... )
    >>> df
       f64  F64  i64  I64  i8  bool_  Bool_   str_
    0  1.1  2.3    2    2   3   True   True  hello

    Select all integer columns (note: booleans are excluded):

    >>> s.select(df, s.integer())
       i64  I64  i8
    0    2    2   3

    Combine with :func:`boolean` to include Boolean columns:

    >>> s.select(df, s.integer() | s.boolean())
       i64  I64  i8  bool_  Bool_
    0    2    2   3   True   True
    """

    return Filter(sbd.is_integer, name="integer")


def float():
    """
    Select columns that have a floating-point data type.

    **When to use:**
    Use this selector when you specifically need floating-point-typed columns,
    excluding integer and Boolean types. This is useful for selecting continuous
    numeric features that have been measured or calculated as decimals.

    This selector selects floating-point columns (float32, float64, etc.).
    Note that ``integer() | float()`` is equivalent to :func:`numeric()`.

    See Also
    --------
    numeric :
        Select all numeric columns (integer and float).
        Use this to select both integer and floating-point columns together.
    integer :
        Select integer columns only.
    boolean :
        Select Boolean columns.

    Examples
    --------
    >>> from skrub import selectors as s
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame(
    ...     dict(
    ...         f64=[1.1],
    ...         F64=pd.Series([2.3]).convert_dtypes(),
    ...         f32=np.asarray(3.4, dtype='float32'),
    ...         i64=[2],
    ...         I64=pd.Series([2]).convert_dtypes(),
    ...         i8=np.int8(3),
    ...         bool_=[True],
    ...         Bool_=pd.Series([True]).convert_dtypes(),
    ...         str_=["hello"],
    ...     )
    ... )
    >>> df
       f64  F64  f32  i64  I64  i8  bool_  Bool_   str_
    0  1.1  2.3  3.4    2    2   3   True   True  hello

    Select all floating-point columns:

    >>> s.select(df, s.float())
       f64  F64  f32
    0  1.1  2.3  3.4

    Combine with other selectors:

    >>> s.select(df, s.float() | s.integer())
       f64  F64  f32  i64  I64  i8
    0  1.1  2.3  3.4    2    2   3

    """
    return Filter(sbd.is_float, name="float")


def _has_dtype(column, *dtypes):
    return sbd.dtype(column) in dtypes


def has_dtype(*dtypes):
    """
    Select columns whose dtype is equal to one of the provided dtypes.

    **When to use:**
    This is an advanced selector for edge cases where you need to match specific
    dtypes not covered by other selectors. Use this when working with specialized
    or custom dtypes (e.g., pandas ListDtype, polars Object).

    For standard types, prefer the simpler selectors like :func:`numeric`,
    :func:`string`, :func:`categorical`, or :func:`boolean`.

    This selector takes a hands-off approach: skrub does not normalize or infer
    dtypes across dataframe libraries. A column is selected if
    ``sbd.dtype(column) == dtype`` for at least one of the provided ``dtypes``.

    Parameters
    ----------
    *dtypes : dtype objects
        One or more dtype objects to match. The most reliable approach is to get
        dtypes from existing columns in your dataframe.

    See Also
    --------
    numeric :
        Select numeric columns (simpler than has_dtype for standard types).
    string :
        Select string columns (simpler than has_dtype for standard types).
    categorical :
        Select categorical columns (simpler than has_dtype for standard types).
    object :
        Select columns with "object" dtype (library specific).

    Notes
    -----
    Some dataframe libraries may accept shorthand values that compare equal to a
    dtype object (e.g., 'int64'), but this is backend-specific. For robustness,
    pass dtype objects obtained from your dataframe library.

    Examples
    --------
    >>> from skrub import selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "items": [["A4", "A3"], ["A5"]],
    ...         "count": [2, 1],
    ...     }
    ... )

    Get dtype from an existing column and use it for selection:

    >>> items_dtype = df["items"].dtype
    >>> s.select(df, s.has_dtype(items_dtype))
           items
    0  [A4, A3]
    1      [A5]

    Match multiple dtypes at once:

    >>> count_dtype = df["count"].dtype
    >>> s.select(df, s.has_dtype(items_dtype, count_dtype))
           items  count
    0  [A4, A3]      2
    1      [A5]      1

    """
    return Filter(_has_dtype, args=dtypes, name="has_dtype")


def any_date():
    """
    Select columns that have a Date or Datetime data type.

    **When to use:**
    Use this selector to find temporal columns that need date-specific
    preprocessing, such as feature extraction (year, month, day) or
    time-based aggregations.

    This selector matches Datetime columns, including timezone-aware datetime
    columns.

    Notes
    -----
    Behavior may differ between pandas and polars:
    - Pandas: Selects datetime64 dtype columns
    - Polars: Selects both Date and Datetime dtypes

    See Also
    --------
    ~skrub.Cleaner :
        Parse and clean date columns into proper datetime types.

    ~skrub.ToDatetime :
        Convert string columns to datetime types.

    Examples
    --------
    >>> import datetime
    >>> from skrub import selectors as s
    >>> import pandas as pd

    >>> df = pd.DataFrame(
    ...     dict(
    ...         dt=[datetime.datetime(2020, 3, 2, 10, 30)],
    ...         tzdt=[
    ...             datetime.datetime(2020, 3, 2, 10, 30, tzinfo=datetime.timezone.utc)
    ...         ],
    ...         str_=["2020-03-02 10:30:00"],
    ...     )
    ... )

    Select all date/datetime columns:

    >>> s.select(df, s.any_date())
                           dt                      tzdt
    0 2020-03-02 10:30:00 2020-03-02 10:30:00+00:00

    Note that string columns with date-like values are not selected
    (use filtering for that):

    >>> s.select(df, s.any_date() | s.string())
                           dt                      tzdt                 str_
    0 2020-03-02 10:30:00 2020-03-02 10:30:00+00:00  2020-03-02 10:30:00

    Consider using :class:`~skrub.Cleaner` or :class:`~skrub.ToDatetime` to
    convert string columns to datetime before selecting them with :func:`any_date`:

    >>> from skrub import Cleaner
    >>> df = pd.DataFrame({"date_str": ["2020-03-02 10:30:00", "2021-05-15 14:45:00"]})
    >>> df = Cleaner().fit_transform(df)
    >>> s.select(df, s.any_date())
                date_str
    0 2020-03-02 10:30:00
    1 2021-05-15 14:45:00


    """
    return Filter(sbd.is_any_date, name="any_date")


def categorical():
    """
    Select columns that have a Categorical (or polars Enum) data type.

    **When to use:**
    Use this selector to find categorical columns with an explicitly defined
    set of categories. This is useful for identifying columns ready for
    encoding or that contain discrete, predefined values.

    Note that categorical columns are different from string columns: categorical
    columns have a fixed set of category values, while string columns contain
    arbitrary text.

    See Also
    --------
    string :
        Select string columns.
        Use this for columns with arbitrary text values.
    cardinality_below :
        Select columns with low cardinality.
        Use this if you have text columns without explicit categories.
    ~skrub.ToCategorical :
        Convert a column to categorical type for explicit category handling.

    Examples
    --------
    >>> from skrub import selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     dict(
    ...         string=pd.Series(['A', 'B']),
    ...         category=pd.Series(['A', 'B'], dtype="category"),
    ...     )
    ... )

    >>> df
      string category
    0      A        A
    1      B        B

    Select only categorical columns (note: string columns are not selected):

    >>> s.select(df, s.categorical())
      category
    0        A
    1        B

    Combine with :func:`string` to select all text-like columns:

    >>> s.select(df, s.categorical() | s.string())
      string category
    0      A        A
    1      B        B

    Use :class:`~skrub.ToCategorical` to convert string columns to categorical
    for selection:

    >>> from skrub import ToCategorical, ApplyToCols
    >>> df = ApplyToCols(ToCategorical(), cols="string").fit_transform(df)
    >>> s.select(df, s.categorical())
    string category
    0      A        A
    1      B        B

    """
    return Filter(sbd.is_categorical, name="categorical")


def string():
    """
    Select columns that have a String data type.

    **When to use:**
    Use this selector to find all text columns for encoding, NLP processing,
    or text-based feature engineering. This includes both explicit string dtypes
    and object columns containing only strings.

    In pandas, object columns containing (only) strings are also selected.

    Notes
    -----
    The behavior of string columns may change depending on the pandas version:

    - Before pandas 3.0: String columns may have the 'object' dtype
    - From pandas 3.0 onwards: String columns have only the 'string' dtype

    This selector handles both cases, selecting string columns regardless of
    pandas version. Object columns containing mixed types (e.g., strings and
    numbers) are not selected.

    See Also
    --------
    categorical :
        Select categorical columns (explicit categories, not arbitrary strings).
    object :
        Select object dtype columns (broader, may include mixed types).
    filter :
        Use for custom text-based selection criteria.

    Examples
    --------

    >>> from skrub import selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     dict(
    ...         object_string=pd.Series(['A', 'B']),
    ...         object=pd.Series(['A', 10]),
    ...         string=pd.Series(['A', 'B']).convert_dtypes(),
    ...         categorical=pd.Series(['A', 'B'], dtype="category"),
    ...     )
    ... )
    >>> df
    object_string object string categorical
    0             A      A      A           A
    1             B     10      B           B

    Select all string columns (note: mixed-type object columns are excluded):

    >>> s.select(df, s.string())
    object_string string
    0             A      A
    1             B      B

    Combine with categorical() to select all text-like columns:

    >>> s.select(df, s.string() | s.categorical())
    object_string string categorical
    0             A      A           A
    1             B      B           B

    """
    return Filter(sbd.is_string, name="string")


def object():
    """
    Select columns whose dtype is ``object`` (pandas) or ``pl.Object`` (polars).

    **When to use:**
    Use this selector to find columns with the object dtype, which may contain
    mixed types. For standard use cases, prefer more specific selectors like
    :func:`string` or :func:`categorical`.

    Notes
    -----
    Before pandas 3.0, columns containing only strings can have the ``object``
    dtype. From pandas 3.0 onwards they have the ``string`` dtype instead.
    Use :func:`string` for selecting string columns across pandas versions.

    The object dtype is broader and less semantic than :func:`string` - it
    can contain any mix of types. Use :func:`object` only when you specifically
    need mixed-type columns.

    See Also
    --------
    string :
        Select string columns (preferred for text data).
        Use this instead of object() for text columns.
    categorical :
        Select categorical columns.
    has_dtype :
        Select columns whose dtype matches specific dtypes.

    Examples
    --------
    >>> from skrub import selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     dict(
    ...         mixed=pd.Series(['A', 10]),
    ...         numeric=pd.Series([1, 2]),
    ...         string=pd.Series(['A', 'B']).convert_dtypes(),
    ...     )
    ... )
    >>> df.dtypes
    mixed       object
    numeric      int64
    string         ...
    dtype: object

    Select object dtype columns (note: may contain mixed types):

    >>> s.select(df, s.object())
      mixed
    0     A
    1    10

    Prefer string() for text columns:

    >>> s.select(df, s.string())
      string
    0      A
    1      B
    """
    return Filter(sbd.is_object, name="object")


def boolean():
    """
    Select columns that have a Boolean data type.

    **When to use:**
    Use this selector to find Boolean columns that often need special handling
    (e.g., encoding strategies different from numeric features). Boolean features
    represent binary choices and may require different preprocessing than numeric
    features.

    Selects columns with bool or boolean dtypes across different dataframe libraries.

    Notes
    -----
    Boolean columns are excluded from :func:`numeric` because they typically
    require different preprocessing strategies than numeric features:

    - Numeric: Usually need scaling/normalization
    - Boolean: Usually need encoding (binary representation)

    See Also
    --------
    numeric :
        Select all numeric columns (integer and float, NOT boolean).
    integer :
        Select integer columns.
    filter :
        Use for custom data-based selection criteria.

    Examples
    --------
    >>> from skrub import selectors as s
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame(
    ...     dict(
    ...         i64=[0],
    ...         i8=np.int8(3),
    ...         bool_=[True],
    ...         Bool_=pd.Series([False]).convert_dtypes(),
    ...     )
    ... )
    >>> df
       i64  i8  bool_  Bool_
    0    0   3   True  False

    Select all Boolean columns:

    >>> s.select(df, s.boolean())
       bool_  Bool_
    0   True  False

    Combine with numeric() to include both (note: boolean() is separate):

    >>> s.select(df, s.boolean() | s.numeric())
       i64  i8  bool_  Bool_
    0    0   3   True  False

    Note that numeric() alone does NOT include Boolean columns:

    >>> s.select(df, s.numeric())
       i64  i8
    0    0   3

    """
    return Filter(sbd.is_bool, name="boolean")


#
# Selectors based on column values, computed statistics
#


def _cardinality_below(column, threshold):
    try:
        return sbd.n_unique(column) < threshold
    except Exception:
        # ``n_unique`` can fail for example for polars columns with dtype Object
        return False


def cardinality_below(threshold):
    """
    Select columns whose cardinality (number of unique values) is (strictly) \
    below ``threshold``.

    **When to use:**
    Use this selector to identify low-cardinality (discrete) features for
    categorical encoding or to find ID-like columns with high cardinality for
    exclusion. This is useful for feature engineering and data quality checks.

    Parameters
    ----------
    threshold : int
        Columns with fewer than this many unique values are selected.
        Null values do not count in the cardinality.

    Notes
    -----
    **Performance Consideration:** This selector requires computing the number
    of unique values for each column. On large datasets (>1M rows), this may
    be slow. Consider using on a sample if performance is critical.

    See Also
    --------
    has_nulls :
        Select columns that contain null values.
    filter :
        Use for custom cardinality-based selection criteria.

    Examples
    --------
    >>> from skrub import selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     dict(
    ...         a1=[1, 1, 1, None],
    ...         a2=[1, 1, 2, None],
    ...         a2_b=[1, 1, 2, 2],
    ...         a3=[1, 2, 3, None],
    ...         a3_b=[1, 2, 3, 3],
    ...         a4=[1, 2, 3, 4],
    ...     )
    ... ).convert_dtypes()
    >>> df
         a1    a2  a2_b    a3  a3_b  a4
    0     1     1     1     1     1   1
    1     1     1     1     2     2   2
    2     1     2     2     3     3   3
    3  <NA>  <NA>     2  <NA>     3   4

    Select low-cardinality columns:

    >>> s.select(df, s.cardinality_below(3))
         a1    a2  a2_b
    0     1     1     1
    1     1     1     1
    2     1     2     2
    3  <NA>  <NA>     2

    Select columns with cardinality below 4:

    >>> s.select(df, s.cardinality_below(4))
         a1    a2  a2_b    a3  a3_b
    0     1     1     1     1     1
    1     1     1     1     2     2
    2     1     2     2     3     3
    3  <NA>  <NA>     2  <NA>     3

    Invert to select high-cardinality columns (e.g., exclude low-cardinality):

    >>> s.select(df, ~s.cardinality_below(3))
        a3  a3_b  a4
    0     1     1   1
    1     2     2   2
    2     3     3   3
    3  <NA>     3   4

    Select numeric features with low cardinality:

    >>> s.select(df, s.cardinality_below(10) & s.numeric())
        a1    a2  a2_b    a3  a3_b  a4
    0     1     1     1     1     1   1
    1     1     1     1     2     2   2
    2     1     2     2     3     3   3
    3  <NA>  <NA>     2  <NA>     3   4

    Note that numeric features are still treated as numeric even if they have low
    cardinality (e.g., IDs): convert them to categorical (e.g., using
    :class:`ToCategorical`) if you want to treat them as categorical features
    instead.

    """
    return Filter(_cardinality_below, args=(threshold,), name="cardinality_below")


def _null_count_check(column, proportion):
    if proportion == 0.0:
        return sbd.has_nulls(column)
    if proportion == 1.0:
        return sbd.is_all_null(column)
    return sum(sbd.is_null(column)) / len(column) > proportion


def has_nulls(proportion=0.0):
    """
    Select columns that contain at least one null value, or a proportion of null \
    values above a given threshold.

    **When to use:**
    Use this selector to identify columns needing imputation or to find columns
    with excessive missing data for exclusion. This is useful for data quality
    checks and preprocessing pipelines.

    Parameters
    ----------
    proportion : float, optional
        Default 0.0. Select columns where the proportion of null values exceeds
        this threshold (range: 0.0 to 1.0).

        - 0.0 (default): Selects any column with at least one null value
        - 0.5: Selects columns with >50% missing values
        - 1.0: Selects columns where all values are null

    Notes
    -----
    Null values include NaN, None, NA, etc., depending on the dataframe library.
    Behavior:

    - pandas: Recognizes np.nan, None, pd.NA, pd.NaT
    - polars: Recognizes null values

    See Also
    --------
    cardinality_below :
        Select columns whose cardinality is below a threshold.
    DropUninformative :
        Automatically drop columns that are uninformative, including columns with
        excessive null values.
    Cleaner :
        Parse common null representations (e.g., 'NA', 'missing') into proper null
        values.
    filter :
        Use for custom null-based selection criteria.

    Examples
    --------
    >>> from skrub import selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(dict(a=[0, 1, 2], b=[0, None, 20], c=['a', 'b', None]))

    Select all columns with at least one null value:

    >>> s.select(df, s.has_nulls())
          b     c
    0   0.0     a
    1   ...     b
    2  20.0  ...

    Select columns with >20% missing values:

    >>> df2 = pd.DataFrame(dict(
    ...     few_nulls=[1, 2, 3, None],
    ...     many_nulls=[1, None, None, None],
    ...     no_nulls=[1, 2, 3, 4]))
    >>> s.select(df2, s.has_nulls(proportion=0.2))
       few_nulls  many_nulls
    0        1.0         1.0
    1        2.0         ...
    2        3.0         ...
    3        ...         ...

    Select columns with >50% missing values:

    >>> s.select(df2, s.has_nulls(proportion=0.5))
    many_nulls
    0        1.0
    1        ...
    2        ...
    3        ...

    Invert to select columns with NO null values:

    >>> s.select(df2, ~s.has_nulls())
    no_nulls
    0        1
    1        2
    2        3
    3        4

    Drop columns with >10% missing data:

    >>> from skrub import DropCols
    >>> DropCols(cols=s.has_nulls(proportion=0.10)).fit_transform(df2)
    no_nulls
    0        1
    1        2
    2        3
    3        4

    Use :class:`Cleaner` to parse common null representations
    into proper null values before selection:

    >>> df3 = pd.DataFrame(dict(
    ...     col1=[1, 2, 3, 'NA'],
    ...     col2=[1, 2, 3, 'missing'],
    ...     col3=[1, 2, 3, 4]))
    >>> from skrub import Cleaner
    >>> df3 = Cleaner(null_strings=['missing']).fit_transform(df3)
    >>> s.select(df3, s.has_nulls())
        col1  col2
    0     1     1
    1     2     2
    2     3     3
    3  None  None
   """

    if not isinstance(proportion, numbers.Number) or not 0.0 <= proportion <= 1.0:
        raise ValueError(
            f"Proportion {proportion} is invalid. Proportion"
            " should be a number in the range [0.0, 1.0]"
        )
    return Filter(_null_count_check, args=(proportion,), name="has_nulls")
