import fnmatch
import re

from .. import _dataframe as sbd
from ._base import Filter, NameFilter

__all__ = [
    "glob",
    "regex",
    "numeric",
    "integer",
    "float",
    "any_date",
    "categorical",
    "string",
    "boolean",
    "cardinality_below",
    "has_nulls",
]

#
# Selectors based on column names
#


def glob(pattern):
    """Select columns by name with Unix shell style 'glob' pattern.

    pattern is interpreted as described in ``fnmatch.fnmatchcase``:

        *       matches everything
        ?       matches any single character
        [seq]   matches any character in seq
        [!seq]  matches any char not in seq

    Examples
    --------
    >>> from skrub import _selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "height_mm": [297.0, 420.0],
    ...         "width_mm": [210.0, 297.0],
    ...         "kind": ["A4", "A3"],
    ...         "ID": [4, 3],
    ...     }
    ... )
    >>> df
       height_mm  width_mm kind  ID
    0      297.0     210.0   A4   4
    1      420.0     297.0   A3   3

    >>> s.select(df, s.glob('*'))
       height_mm  width_mm kind  ID
    0      297.0     210.0   A4   4
    1      420.0     297.0   A3   3
    >>> s.select(df, s.glob('*_mm'))
       height_mm  width_mm
    0      297.0     210.0
    1      420.0     297.0

    """
    return NameFilter(fnmatch.fnmatchcase, args=(pattern,), name="glob")


def _regex(col_name, pattern, flags=0):
    return re.match(pattern, col_name, flags=flags) is not None


def regex(pattern, flags=0):
    """Select columns by name with a regular expression.

    pattern can be a string pattern or a compiled regular expression, and flags
    are regular expression flags as described in the ``re`` module
    documentation:

    https://docs.python.org/3/library/re.html#flags

    Examples
    --------
    >>> from skrub import _selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "height_mm": [297.0, 420.0],
    ...         "width_mm": [210.0, 297.0],
    ...         "kind": ["A4", "A3"],
    ...         "ID": [4, 3],
    ...     }
    ... )
    >>> df
       height_mm  width_mm kind  ID
    0      297.0     210.0   A4   4
    1      420.0     297.0   A3   3

    >>> s.select(df, s.regex('.*_mm'))
       height_mm  width_mm
    0      297.0     210.0
    1      420.0     297.0

    A column is selected if ``re.match(col_name, pattern, flags)`` returns a
    match. Note that it is enough to match at the beginning of the string:

    >>> s.select(df, s.regex('wid'))
       width_mm
    0     210.0
    1     297.0

    Use '$' to require matching until the end of the column name:

    >>> s.select(df, s.regex('wid$'))
    Empty DataFrame
    Columns: []
    Index: [0, 1]

    Flags are passed to ``re.match``; the following are 3 equivalent ways of
    setting re flags (re.IGNORECASE in this example):

    >>> import re
    >>> s.select(df, s.regex('id', flags=re.I))
       ID
    0   4
    1   3
    >>> s.select(df, s.regex('(?i)id'))
       ID
    0   4
    1   3
    >>> s.select(df, s.regex(re.compile('id', re.I)))
       ID
    0   4
    1   3

    """
    kwargs = {"flags": flags} if flags != 0 else {}
    return NameFilter(_regex, args=(pattern,), kwargs=kwargs, name="regex")


#
# Selectors based on data types
#


def numeric():
    """
    Select columns that have a numeric data type.

    This selects float and integer columns but not Boolean columns.

    Examples
    --------
    >>> from skrub import _selectors as s
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
    str_      object
    dtype: object

    >>> s.select(df, s.numeric())
       f64  F64  i64  I64  i8
    0  1.1  2.3    2    2   3

    Use s.boolean() to also select Boolean columns:

    >>> s.select(df, s.numeric() | s.boolean())
       f64  F64  i64  I64  i8  bool_  Bool_
    0  1.1  2.3    2    2   3   True   True

    """
    return Filter(sbd.is_numeric, name="numeric")


def integer():
    """
    Select columns that have an integer data type.

    This selects integer columns but not Boolean columns.

    Examples
    --------
    >>> from skrub import _selectors as s
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
    str_      object
    dtype: object

    >>> s.select(df, s.integer())
       i64  I64  i8
    0    2    2   3

    Use s.boolean() to also select Boolean columns:

    >>> s.select(df, s.integer() | s.boolean())
       i64  I64  i8  bool_  Bool_
    0    2    2   3   True   True
    """

    return Filter(sbd.is_integer, name="integer")


def float():
    """
    Select columns that have a floating-point data type.

    Examples
    --------
    >>> from skrub import _selectors as s
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
    str_      object
    dtype: object

    >>> s.select(df, s.float())
       f64  F64  f32
    0  1.1  2.3  3.4

    """
    return Filter(sbd.is_float, name="float")


def any_date():
    """
    Select columns that have a Date or Datetime data type.

    Examples
    --------
    >>> import datetime
    >>> from skrub import _selectors as s
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
    dt           datetime64[ns]
    tzdt    datetime64[ns, UTC]
    str_                 object
    dtype: object

    >>> s.select(df, s.any_date())
                       dt                      tzdt
    0 2020-03-02 10:30:00 2020-03-02 10:30:00+00:00

    """
    return Filter(sbd.is_any_date, name="any_date")


def categorical():
    """
    Select columns that have a Categorical (or polars Enum) data type.

    Examples
    --------
    >>> from skrub import _selectors as s
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

    >>> df.dtypes
    string        object
    category    category
    dtype: object

    >>> s.select(df, s.categorical())
      category
    0        A
    1        B

    """
    return Filter(sbd.is_categorical, name="categorical")


def string():
    """
    Select columns that have a String data type.

    In pandas, object columns containing (only) strings are also selected.

    Examples
    --------

    >>> from skrub import _selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     dict(
    ...         os=pd.Series(['A', 'B']),
    ...         o=pd.Series(['A', 10]),
    ...         s=pd.Series(['A', 'B']).convert_dtypes(),
    ...         c=pd.Series(['A', 'B'], dtype="category"),
    ...     )
    ... )
    >>> df
      os   o  s  c
    0  A   A  A  A
    1  B  10  B  B

    >>> df.dtypes
    os            object
    o             object
    s     string...
    c           category
    dtype: object

    >>> s.select(df, s.string())
      os  s
    0  A  A
    1  B  B

    >>> s.select(df, s.string() | s.categorical())
      os  s  c
    0  A  A  A
    1  B  B  B

    """
    return Filter(sbd.is_string, name="string")


def boolean():
    """
    Select columns that have an Boolean data type.

    Examples
    --------
    >>> from skrub import _selectors as s
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
    >>> df.dtypes
    i64        int64
    i8          int8
    bool_       bool
    Bool_    boolean
    dtype: object

    >>> s.select(df, s.boolean())
       bool_  Bool_
    0   True  False

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
    Select columns whose cardinality (number of unique values) is (strictly)
    below ``threshold``.

    Null values do not count in the cardinality.

    Examples
    --------
    >>> from skrub import _selectors as s
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
    >>> s.select(df, s.cardinality_below(3))
         a1    a2  a2_b
    0     1     1     1
    1     1     1     1
    2     1     2     2
    3  <NA>  <NA>     2
    >>> s.select(df, s.cardinality_below(4))
         a1    a2  a2_b    a3  a3_b
    0     1     1     1     1     1
    1     1     1     1     2     2
    2     1     2     2     3     3
    3  <NA>  <NA>     2  <NA>     3

    """
    return Filter(_cardinality_below, args=(threshold,), name="cardinality_below")


def has_nulls():
    """
    Select columns that contain at least one null value.

    Examples
    --------
    >>> from skrub import _selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(dict(a=[0, 1, 2], b=[0, None, 20], c=['a', 'b', None]))
    >>> s.select(df, s.has_nulls())
          b     c
    0   0.0     a
    1   NaN     b
    2  20.0  None
    """
    return Filter(sbd.has_nulls, name="has_nulls")
