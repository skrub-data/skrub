.. _userguide_selectors:

Skrub Selectors: helpers for selecting columns in a dataframe
=============================================================

In Skrub, a selector represents a column selection rule, such as "all columns that have
numerical data types, except the column ``'User ID'``".

Selectors have two main benefits:

- Expressing complex selection rules in a simple and concise way by combining selectors
  with operators. A range of useful selectors is provided by this module.
- Delayed selection: passing a selection rule which will evaluated later on a dataframe
  that is not yet available. For example, without selectors, it is not possible to
  instantiate a :class:`~skrub.SelectCols` that selects all columns except those with
  the suffix 'ID' if the data on which it will be fitted is not yet available.

Usage
-----

Here is an example dataframe. Note that selectors support both Pandas and Polars
dataframes:

>>> import pandas as pd
>>> df = pd.DataFrame(
...     {
...         "height_mm": [297.0, 420.0],
...         "width_mm": [210.0, 297.0],
...         "kind": ["A4", "A3"],
...         "ID": [4, 3],
...     }
... )

:func:`~skrub.selectors.cols` is a simple kind of selector which selects a fixed list of
column names:

>>> from skrub import selectors as s
>>> mm_cols = s.cols('height_mm', 'width_mm')
>>> mm_cols
cols('height_mm', 'width_mm')

This selector can then be passed to a :func:`~skrub.selectors.select` function:

>>> s.select(df, mm_cols)
   height_mm  width_mm
0      297.0     210.0
1      420.0     297.0

It can also be passed to :class:`~skrub.SelectCols` or :class:`~skrub.DropCols`
to be embedded in scikit-learn pipelines:

>>> from skrub import SelectCols
>>> SelectCols(cols=mm_cols).fit_transform(df)
   height_mm  width_mm
0      297.0     210.0
1      420.0     297.0

Last but not least, selectors can be passed to skrub DataOps when applying an
estimator with the :func:`skrub.DataOp.skb.apply` function:

>>> import skrub
>>> from sklearn.preprocessing import StandardScaler
>>> skrub.X(df).skb.apply(StandardScaler(), cols=mm_cols)
<Apply StandardScaler>
Result:
―――――――
  kind  ID  height_mm  width_mm
0   A4   4       -1.0      -1.0
1   A3   3        1.0       1.0


Type of selectors
-----------------

:func:`~skrub.selectors.all` is another simple selector, especially useful for default
arguments since it keeps all columns:

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

>>> SelectCols(s.glob('*_mm')).fit_transform(df)
   height_mm  width_mm
0      297.0     210.0
1      420.0     297.0

See :ref:`selectors_ref` for an exhaustive list.

The available operators are ``|``, ``&``, ``-``, ``^`` with the meaning of usual
python sets, and ``~`` to invert a selection.

>>> SelectCols(s.glob('*_mm')).fit_transform(df)
   height_mm  width_mm
0      297.0     210.0
1      420.0     297.0

>>> SelectCols(~s.glob('*_mm')).fit_transform(df)
  kind  ID
0   A4   4
1   A3   3

>>> SelectCols(s.glob('*_mm') | s.cols('ID')).fit_transform(df)
   height_mm  width_mm  ID
0      297.0     210.0   4
1      420.0     297.0   3

>>> SelectCols(s.glob('*_mm') & s.glob('height_*')).fit_transform(df)
   height_mm
0      297.0
1      420.0

>>> SelectCols(s.glob('*_mm') ^ s.string()).fit_transform(df)
   height_mm  width_mm kind
0      297.0     210.0   A4
1      420.0     297.0   A3

The operators respect the usual short-circuit rules. For example, the
following selector won't compute the cardinality of non-categorical columns:

>>> s.categorical() & s.cardinality_below(10)
(categorical() & cardinality_below(10))


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
