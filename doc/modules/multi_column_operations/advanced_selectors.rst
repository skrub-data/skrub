.. currentmodule :: skrub.selectors

.. |ApplyToCols| replace:: :class:`~skrub.ApplyToCols`
.. |StandardScaler| replace:: :class:`~sklearn.preprocessing.StandardScaler`
.. |filter| replace:: :func:`filter`
.. |filter_names| replace:: :func:`filter_names`

.. _user_guide_advanced_selectors:

Advanced selectors: |filter| and |filter_names|
-------------------------------------------

:func:`filter` and :func:`filter_names` allow
selecting columns based on arbitrary user-defined criteria. These are also used to
implement many of the other selectors provided in this module.

:func:`filter` accepts a function which will be called on a column
(i.e., a Pandas or polars Series). This function, called a predicate, must return
``True`` if the column should be selected.

>>> import pandas as pd
>>> import skrub.selectors as s
>>> df = pd.DataFrame(
...     {
...         "height_mm": [297.0, 420.0],
...         "width_mm": [210.0, 297.0],
...         "kind": ["A4", "A3"],
...         "ID": [4, 3],
...     }
... )
>>> s.select(df, s.filter(lambda col: "A4" in col.tolist()))
  kind
0   A4
1   A3

:func:`filter_names` accepts a predicate that is passed the column name,
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

Combining selectors with other skrub transformers
-------------------------------------------------
Skrub transformers are designed to be used in conjunction with other transformers
that operate on columns to improve their versatility.

For example, we can drop columns that have more unique values than a certain amount
by combining :func:`cardinality_below` with :class:`skrub.DropCols`.
We first select the columns that have more than 3 unique values, then we invert the
selector and finally transform the dataframe.

>>> df = pd.DataFrame({
... "not a lot": [1, 1, 1, 2, 2],
... "too_many":  [1,2,3,4,5]})

>>> from skrub import DropCols
>>> DropCols(cols=~s.cardinality_below(3)).fit_transform(df)
   not a lot
0          1
1          1
2          1
3          2
4          2

Selectors can be used in conjunction with |ApplyToCols| to transform columns
based on specific requirements.

Consider the following example:

>>> import pandas as pd
>>> data = {
...     "subject": ["Math", "English", "History", "Science", "Art"],
...     "grade": [5, 4, 3, 4, 3]
... }
>>> df = pd.DataFrame(data)
>>> df
   subject grade
0     Math     5
1  English     4
2  History     3
3  Science     4
4      Art     3

We might want to apply the |StandardScaler| only to the numeric column. We can
do this like this:

>>> from skrub import ApplyToCols
>>> from sklearn.preprocessing import StandardScaler
>>> ApplyToCols(StandardScaler(), cols=s.numeric()).fit_transform(df)
   subject     grade
0     Math  1.603567
1  English  0.267261
2  History -1.069045
3  Science  0.267261
4      Art -1.069045
