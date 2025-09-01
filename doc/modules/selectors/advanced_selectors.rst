Advanced selectors: filter and filter_names
-------------------------------------------

:func:`skrub.selectors.filter` and :func:`skrub.selectors.filter_names` allow
selecting columns based on arbitrary user-defined criteria. These are also used to
implement many of the other selectors provided in this module.

:func:`skrub.selectors.filter` accepts a function which will be called on a column
(i.e., a Pandas or polars Series). This function, called a predicate, must return
``True`` if the column should be selected.

>>> s.select(df, s.filter(lambda col: "A4" in col.tolist()))
  kind
0   A4
1   A3

:func:`skrub.selectors.filter_names` accepts a predicate that is passed the column name,
instead of the column.

>>> s.select(df, s.filter_names(lambda name: name.endswith('mm')))
   height_mm  width_mm
0      297.0     210.0
1      420.0     297.0

We can pass args and kwargs that will be forwarded to the predicate, to help avoid
lambda or local functions and thus ensure the selector is picklable.

>>> s.select(df, s.filter_names(str.endswith, 'mm'))
   height_mm  width_mm
0      297.0     210.0
1      420.0     297.0
