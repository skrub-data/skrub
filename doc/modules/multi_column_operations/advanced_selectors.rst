.. currentmodule:: skrub.selectors

.. |ApplyToCols| replace:: :class:`~skrub.ApplyToCols`
.. |StandardScaler| replace:: :class:`~sklearn.preprocessing.StandardScaler`
.. |filter| replace:: :func:`filter`
.. |filter_names| replace:: :func:`filter_names`

.. _user_guide_advanced_selectors:

|filter| and |filter_names| to select with user-defined criteria
-----------------------------------------------------------------

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


Example of custom criteria in :func:`filter`: selecting columns with outliers
.............................................................................

The :func:`filter` selector can be used to select columns based on custom
criteria. For example, we can define a function that checks if a column contains
outliers using the Interquartile Range (IQR) method, and then use this function
with :func:`filter` to select such columns.

Specifically, we define a function that computes the IQR (Inter Quartile Range) of a column
and checks if any data points extend further than 2 IQRs of the lower and upper quartile.

>>> def has_outliers(column):
...    q1 = column.quantile(0.25)
...    q3 = column.quantile(0.75)
...    IQR = q3 - q1
...    lower_bound = q1 - 2 * IQR
...    upper_bound = q3 + 2 * IQR
...    outliers = (column < lower_bound) | (column > upper_bound)
...    return any(outliers)

>>> from skrub import SelectCols
>>> select = SelectCols(s.filter(has_outliers))
>>> data = pd.DataFrame({
...     "A": [10, 12, 14, 15, 100],  # Outlier in column A
...     "B": [20, 22, 21, 19, 20],   # No outliers in column B
...     "C": [30, 29, 31, 32, 300]   # Outlier in column C
... })
>>> select.fit_transform(data)
     A    C
0   10   30
1   12   29
2   14   31
3   15   32
4  100  300
