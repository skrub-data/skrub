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


Select columns with null values
--------------------------------
Selectors :func:`has_nulls` and :func:`DropUninformative` can be used to get information
about columns with null values. The selector :func:`has_nulls` selects columns that contain
null values and it accepts an optional ``threshold`` parameter that allows **selecting** columns
based on the proportion of null values they contain. Similarly, :func:`DropUninformative`
accepts the optional parameter ``drop_null_fraction`` that allows **dropping** columns with a
proportion of null values above the given threshold.

Example: Selecting columns by null percentage with :func:`has_nulls`
.....................................................................

The :func:`has_nulls` selector can filter columns based on their proportion of missing values.
This is useful for identifying columns that may need imputation or further investigation.

>>> import pandas as pd
>>> import skrub.selectors as s
>>> from skrub import SelectCols

Create a dataset with varying amounts of missing data:

>>> df = pd.DataFrame({
...     'patient_id': [1, 2, 3, 4, 5, 6, 7, 8],
...     'age': [25.0, 30.0, None, 45.0, 50.0, None, 60.0, 65.0],           # 25% nulls
...     'blood_pressure': [120, None, None, None, 140, None, None, 150],  # 62.5% nulls
...     'diagnosis': ['flu', 'cold', None, None, None, None, None, None], # 75% nulls
...     'treatment': ['med_A', 'med_B', 'med_C', 'med_D', 'med_E', 'med_F', 'med_G', 'med_H']  # no nulls
... })

Select columns with at least 25% missing values:

>>> s.select(df, s.has_nulls(threshold=0.25))
    age  blood_pressure diagnosis
0  25.0           120.0       flu
1  30.0             NaN      cold
2   NaN             NaN       NaN
3  45.0             NaN       NaN
4  50.0           140.0       NaN
5   NaN             NaN       NaN
6  60.0             NaN       NaN
7  65.0           150.0       NaN

Example: Dropping columns with :func:`DropUninformative`
..........................................................

While :func:`has_nulls` **selects** columns with nulls, :func:`DropUninformative` does the
opposite, it **drops** Columns. This is useful in data cleaning pipelines.

>>> from skrub import DropUninformative, ApplyToCols

Using the same medical dataset, we can drop columns with more than 50% missing values:

>>> cleaner = ApplyToCols(DropUninformative(drop_null_fraction=0.5))
>>> cleaned_df = cleaner.fit_transform(df)
>>> cleaned_df
   patient_id   age treatment
0           1  25.0     med_A
1           2  30.0     med_B
2           3   ...     med_C
3           4  45.0     med_D
4           5  50.0     med_E
5           6   ...     med_F
6           7  60.0     med_G
7           8  65.0     med_H

Example: Creating missing indicators with :func:`has_nulls` and :class:`MissingIndicator`
...........................................................................................

You can combine :func:`has_nulls` with scikit-learn's :class:`~sklearn.impute.MissingIndicator`
to create binary indicator columns for missing values. This is useful when the fact that
a value is missing might be informative for your model.

>>> from sklearn.impute import MissingIndicator

Using the medical dataset, create missing indicators only for columns with at least 25% nulls:

>>> missing_indicator = ApplyToCols(
...     MissingIndicator(),
...     cols=s.has_nulls(threshold=0.25)
... )
>>> indicators = missing_indicator.fit_transform(df)

The original values are:
>>> s.select(df, s.has_nulls(threshold=0.25))
    age  blood_pressure diagnosis
0  25.0           120.0       flu
1  30.0             NaN      cold
2   NaN             NaN       NaN
3  45.0             NaN       NaN
4  50.0           140.0       NaN
5   NaN             NaN       NaN
6  60.0             NaN       NaN
7  65.0           150.0       NaN

After applying the missing indicator transformer, we get:
>>> indicators
   age  blood_pressure  diagnosis
0    0               0          0
1    0               1          0
2    1               1          1
3    0               1          1
4    0               0          1
5    1               1          1
6    0               1          1
7    0               0          1

The indicator columns show where values were missing (1) or present (0). Notice that
only columns with ≥25% nulls were processed: 'age', 'blood_pressure', and 'diagnosis'.
The 'patient_id' and 'treatment' columns were not included because they have no nulls.
