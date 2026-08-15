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

    Use this selector for complex name patterns that glob patterns cannot express.
    This is useful for selecting columns with specific naming conventions or
    patterns that glob patterns cannot express, so that regular expressions are
    needed (e.g., columns matching ``'^feature_[0-9]+$'``).
    For simple wildcard patterns, consider :func:`glob`.

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

    Use regex flags for case-insensitive matching (refer to the regex docs for
    more detail):

    >>> import re
    >>> s.select(df, s.regex('id', flags=re.I))
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
    """Select columns that have a numeric data type.

    Numeric columns include both integer and floating-point types,
    but exclude Boolean columns.

    This selector matches both integer and floating-point columns, equivalent to
    ``integer() | float()``.

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
    >>> df.dtypes
    f64      float64
    F64      Float64
    i64        int64
    I64        Int64
    i8          int8
    bool_       bool
    Bool_    boolean
    str_      ...
    dtype: object

    Select all numeric columns:

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
    """Select columns that have an integer data type.

    Boolean columns are not matched by this selector, only signed and unsigned
    ints are.

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
    >>> df.dtypes
    f64      float64
    F64      Float64
    i64        int64
    I64        Int64
    i8          int8
    bool_       bool
    Bool_    boolean
    str_      ...
    dtype: object

    Select all integer columns:

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
    """Select columns that have a floating-point data type (float32, float64, etc.)

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
    >>> df.dtypes
    f64      float64
    F64      Float64
    f32      float32
    i64        int64
    I64        Int64
    i8          int8
    bool_       bool
    Bool_    boolean
    str_      ...
    dtype: object

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
    """Select columns whose dtype is equal to one of the provided dtypes.

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
        One or more dtype objects to match.

    See Also
    --------
    numeric :
        Select numeric columns.
    string :
        Select string columns.
    categorical :
        Select categorical columns.
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

    >>> s.select(df, s.has_dtype(df["items"].dtype))
           items
    0  [A4, A3]
    1      [A5]

    Match multiple dtypes at once:

    >>> items_dtype = df["items"].dtype
    >>> count_dtype = df["count"].dtype
    >>> s.select(df, s.has_dtype(items_dtype, count_dtype))
           items  count
    0  [A4, A3]      2
    1      [A5]      1

    This also works with complex dtypes such as GeometryDtype in GeoPandas:

    >>> import geopandas as gpd # doctest: +SKIP
    >>> from shapely.geometry import Point # doctest: +SKIP
    >>> gdf = gpd.GeoDataFrame( # doctest: +SKIP
    ...     {"city": ["Paris", "Berlin"], "value": [1, 2]},
    ...     geometry=[Point(2.35, 48.86), Point(13.40, 52.52)],
    ...     crs="EPSG:4326",
    ... )
    >>> gdf # doctest: +SKIP
        city  value            geometry
    0   Paris      1  POINT (2.35 48.86)
    1  Berlin      2  POINT (13.4 52.52)
    >>> s.select(gdf, s.has_dtype(gdf.geometry.dtype)) # doctest: +SKIP
                geometry
    0  POINT (2.35 48.86)
    1  POINT (13.4 52.52)

    """
    return Filter(_has_dtype, args=dtypes, name="has_dtype")


def any_date():
    """Select columns that have a Date or Datetime data type.

    Notes
    -----
    Only datetime columns are selected. Time-only, period, and duration types are
    not selected.
    Selection is based on the column's dtype: for example string columns containing
    date-like values are not selected.

    Selected columns depend on the dataframe library and its supported dtypes:
    in pandas, this selector selects columns with dtype ``datetime64[ns]``,
    while in polars, it selects both ``Date`` and ``Datetime`` dtypes.

    See Also
    --------
    skrub.Cleaner :
        Parse and clean date columns into proper datetime types.

    skrub.ToDatetime :
        Convert string columns to datetime types.

    skrub.DatetimeEncoder :
        Encode datetime columns into numeric features for machine learning.

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
    >>> df
                       dt                      tzdt                 str_
    0 2020-03-02 10:30:00 2020-03-02 10:30:00+00:00  2020-03-02 10:30:00

    >>> df.dtypes
    dt           datetime64[...]
    tzdt    datetime64[..., UTC]
    str_                     ...
    dtype: object

    Select all date/datetime columns:

    >>> s.select(df, s.any_date())
                           dt                      tzdt
    0 2020-03-02 10:30:00 2020-03-02 10:30:00+00:00

    Note that string columns with date-like values are not selected
    (use filtering for that):

    >>> s.select(df, s.any_date() | s.string())
                           dt                      tzdt                 str_
    0 2020-03-02 10:30:00 2020-03-02 10:30:00+00:00  2020-03-02 10:30:00

    """
    return Filter(sbd.is_any_date, name="any_date")


def categorical():
    """Select columns that have a Categorical (or polars Enum) data type.

    See Also
    --------
    string :
        Select string columns.
    cardinality_below :
        Select columns with low cardinality (low number of unique values).
    skrub.ToCategorical :
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

    """
    return Filter(sbd.is_categorical, name="categorical")


def string():
    """Select columns that have a string data type.

    In pandas, object columns containing strings are also selected.

    Notes
    -----

    .. warning::

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
    """Select columns whose dtype is ``object`` (pandas) or ``pl.Object`` (polars).

    Note that object columns may contain mixed types (e.g., strings and numbers) and are
    broader than string columns. Use this selector when you specifically need
    object-typed columns, and prefer more specific selectors like :func:`string`
    or :func:`categorical`.

    Notes
    -----

    .. warning::

      The behavior of string columns may change depending on the pandas version:

      - Before pandas 3.0: String columns may have the ``object`` dtype
      - From pandas 3.0 onwards: String columns have only the ``string`` dtype

    This selector selects **all** ``object`` dtype columns regardless of content,
    including mixed-type columns. For text data, prefer :func:`string` which is
    more selective.

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

    Select object dtype columns (note: can contain mixed types):

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
    """Select columns that have a Boolean data type.

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

    Combine with numeric() to include both:

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
    """Select columns whose cardinality (number of unique values) is (strictly) \
    below ``threshold``.

    This selector is useful for identifying low-cardinality (discrete) features for
    categorical encoding or for finding ID-like columns with high cardinality to
    encode them in specific ways.

    Parameters
    ----------
    threshold : int
        Columns with fewer than this many unique values are selected.
        Null values do not count in the cardinality.

    Notes
    -----
    Missing values do not count as unique values for cardinality. For example,
    a column with values `[1, 2, 2, None]` has a cardinality of 2.

    If unique value counting fails for a column (e.g., due to unsupported data types),
    the column is not selected.

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

    Select low-cardinality columns (e.g., below 3 unique values):

    >>> s.select(df, s.cardinality_below(3))
         a1    a2  a2_b
    0     1     1     1
    1     1     1     1
    2     1     2     2
    3  <NA>  <NA>     2

    Invert to select high-cardinality columns (i.e., exclude low-cardinality):

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

    """
    return Filter(_cardinality_below, args=(threshold,), name="cardinality_below")


def _null_count_check(column, proportion):
    if proportion == 0.0:
        return sbd.has_nulls(column)
    if proportion == 1.0:
        return sbd.is_all_null(column)
    return sum(sbd.is_null(column)) / len(column) > proportion


def has_nulls(proportion=0.0):
    """Select columns that contain at least one null value, or a proportion of null \
    values above a given threshold.

    Use this selector to identify columns needing imputation or
    with excessive missing data. This is useful for data quality
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
    - polars: Recognizes null values, and NaNs are treated as nulls.

    See Also
    --------
    cardinality_below :
        Select columns whose cardinality is below a threshold.
    skrub.DropUninformative :
        Automatically drop columns that are uninformative, including columns with
        more null values than a specified threshold.
    skrub.Cleaner :
        Parse common null representations (e.g., 'NA', 'missing') into proper null
        values, and possibly drop columns with excessive nulls.
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
    2  20.0   ...

    Select columns with >50% missing values:

    >>> df2 = pd.DataFrame(dict(
    ...     few_nulls=[1, 2, 3, None],
    ...     many_nulls=[1, None, None, None],
    ...     no_nulls=[1, 2, 3, 4]))

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

   """

    if not isinstance(proportion, numbers.Number) or not 0.0 <= proportion <= 1.0:
        raise ValueError(
            f"Proportion {proportion} is invalid. Proportion"
            " should be a number in the range [0.0, 1.0]"
        )
    return Filter(_null_count_check, args=(proportion,), name="has_nulls")
