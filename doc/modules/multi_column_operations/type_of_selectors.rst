Selecting based on dtype or data properties
-------------------------------------------
Selectors can filter columns based on different conditions.

:func:`~skrub.selectors.all` is a simple selector, especially useful for default
arguments since it keeps all columns:

>>> import pandas as pd
>>> from skrub import SelectCols
>>> import skrub.selectors as s
>>> df = pd.DataFrame(
...     {
...         "height_mm": [297.0, 420.0],
...         "width_mm": [210.0, 297.0],
...         "kind": ["A4", "A3"],
...         "ID": [4, 3],
...     }
... )
>>> SelectCols(cols=s.all()).fit_transform(df)
   height_mm  width_mm kind  ID
0      297.0     210.0   A4   4
1      420.0     297.0   A3   3

Selectors can be combined with operators, for example if we wanted all columns
except the "mm" columns above:

>>> SelectCols(s.all() - s.cols("height_mm", "width_mm")).fit_transform(df)
  kind  ID
0   A4   4
1   A3   3

This module provides several kinds of selectors, which allow to select columns by
name, data type, contents, or according to arbitrary user-provided rules.

>>> SelectCols(s.numeric()).fit_transform(df)
   height_mm  width_mm  ID
0      297.0     210.0   4
1      420.0     297.0   3

Selectors can be inverted with ``~``, or :func:`~skrub.selectors.inv`:

>>> SelectCols(~s.numeric()).fit_transform(df)
  kind
0   A4
1   A3

>>> SelectCols(s.inv(s.numeric())).fit_transform(df)
  kind
0   A4
1   A3


Selectors can work on the column names. For example, to select the columns that
end with ``_mm`` we can do:

>>> SelectCols(s.glob('*_mm')).fit_transform(df)
   height_mm  width_mm
0      297.0     210.0
1      420.0     297.0

|

Categories of selectors
-----------------------

The selectors in this module can be categorized based on what aspect of the columns
they examine:

Selectors based on column data types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`~skrub.selectors.numeric`: Select columns with numeric data types (float and integer)
- :func:`~skrub.selectors.integer`: Select columns with integer data types
- :func:`~skrub.selectors.float`: Select columns with floating-point data types
- :func:`~skrub.selectors.any_date`: Select columns with date or datetime data types
- :func:`~skrub.selectors.categorical`: Select columns with categorical data types
- :func:`~skrub.selectors.string`: Select columns with string data types
- :func:`~skrub.selectors.boolean`: Select columns with boolean data types

Selectors based on column content and properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`~skrub.selectors.cardinality_below`: Select columns with fewer unique
  values than a threshold
- :func:`~skrub.selectors.has_nulls`: Select columns that contain at least one
  null value

Selectors based on column names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`~skrub.selectors.cols`: Select columns explicitly by name
- :func:`~skrub.selectors.glob`: Select columns by name using Unix shell-style
  pattern matching
- :func:`~skrub.selectors.regex`: Select columns by name using regular expressions
