.. |DropUninformative| replace:: :class:`~skrub.DropUninformative`
.. |ApplyToEachCol| replace:: :class:`~skrub.ApplyToEachCol`
.. |Cleaner| replace:: :class:`~skrub.Cleaner`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`

.. _user_guide_drop_uninformative:

Removing unneeded columns with |DropUninformative| and |Cleaner|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data tables often include columns that do not provide meaningful information.
These columns increase computational cost and may reduce downstream performance.

The |DropUninformative| transformer removes features that are deemed "uninformative"
using various heuristics. These heuristics include:

- **Dropping columns with excessive missing values**: Columns are dropped if the
  fraction of missing values exceeds the specified threshold. By default, the
  threshold is 1, meaning only columns with all missing values are dropped. Adjust
  this behavior by setting the ``drop_null_fraction`` parameter. Setting it to
  ``None`` disables this check entirely.

- **Dropping constant columns**: Columns containing only a single unique value are
  removed. This behavior is controlled by the ``drop_if_constant`` parameter, which
  is set to ``False`` by default. Note that missing values are treated as distinct
  values, so constant columns with missing values will not be dropped.

- **Dropping unique string/categorical columns**: Columns where each row has a
  unique value (e.g., alphanumeric IDs) can be dropped. This is controlled by the
  ``drop_if_unique`` parameter, which is ``False`` by default. Be cautious when
  enabling this option, as it may remove columns containing free-flowing text.

|DropUninformative| is used by both |TableVectorizer| and |Cleaner|, and both
accept the same parameters for dropping columns.

Consider the following example:

>>> import numpy as np
>>> import pandas as pd
>>> from skrub import Cleaner
>>> data = {
...     'Const int': [1, 1, 1],  # Single unique value
...     'B': [2, 3, 2],  # Multiple unique values
...     'Const str': ['x', 'x', 'x'],  # Single unique value
...     'D': [4, 5, 6],  # Multiple unique values
...     'All nan': [np.nan, np.nan, np.nan],  # All missing values
...     'All empty': ['', '', ''],  # All empty strings
... }
>>> df = pd.DataFrame(data)
>>> df
   Const int  B Const str  D  All nan All empty
0          1  2         x  4      NaN
1          1  3         x  5      NaN
2          1  2         x  6      NaN

To drop constant columns and those with only single values:

>>> cleaner = Cleaner(drop_if_constant=True)
>>> df_cleaned = cleaner.fit_transform(df)
>>> df_cleaned
   B  D
0  2  4
1  3  5
2  2  6

|
Dropping columns with many missing values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Columns with too many missing values may not provide useful information for
downstream models. The ``drop_null_fraction`` parameter allows dropping such
columns when the proportion of missing values exceeds a specified threshold.

Consider the following dataset:

>>> import pandas as pd
>>> from skrub import DropUninformative, ApplyToCols

>>> df = pd.DataFrame({
...     'patient_id': [1, 2, 3, 4, 5, 6, 7, 8],
...     'age': [25.0, 30.0, None, 45.0, 50.0, None, 60.0, 65.0],
...     'blood_pressure': [120, None, None, None, 140, None, None, 150],
...     'diagnosis': ['flu', 'cold', None, None, None, None, None, None],
...     'treatment': ['med_A', 'med_B', 'med_C', 'med_D', 'med_E', 'med_F', 'med_G', 'med_H']
... })

We can drop columns with more than 50% missing values:

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



Applying |DropUninformative| only to a subset of columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can apply the |DropUninformative| transformer to specific columns using
|ApplyToEachCol|.

>>> from skrub import ApplyToEachCol
>>> df = pd.DataFrame({
... "id_to_drop": ["A1", "A2", "A3"],
... "text_to_keep": ["foo", "bar", "baz"]
... })
>>> df
  id_to_drop text_to_keep
0         A1          foo
1         A2          bar
2         A3          baz

Dropping unique columns in this dataframe results in an empty dataframe:

>>> cleaner = Cleaner(drop_if_unique=True)
>>> cleaner.fit_transform(df)
Empty DataFrame
Columns: []
Index: [0, 1, 2]

To apply the transformer only to the ``id_to_drop`` column, use |ApplyToEachCol|:

>>> ApplyToEachCol(cleaner, cols="id_to_drop")
ApplyToEachCol(cols='id_to_drop', transformer=Cleaner(drop_if_unique=True))
>>> ApplyToEachCol(cleaner, cols="id_to_drop").fit_transform(df)
  text_to_keep
0          foo
1          bar
2          baz

For more advanced filtering operations, refer to the User Guide on
:ref:`user_guide_selectors` and the |ApplyToEachCol| documentation for details
on applying transformers to specific columns.
