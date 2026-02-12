.. |DropUninformative| replace:: :class:`~skrub.DropUninformative`
.. |ApplyToCols| replace:: :class:`~skrub.ApplyToCols`
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

Applying |DropUninformative| only to a subset of columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can apply the |DropUninformative| transformer to specific columns using
|ApplyToCols|.

>>> from skrub import ApplyToCols
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

To apply the transformer only to the ``id_to_drop`` column, use |ApplyToCols|:

>>> ApplyToCols(cleaner, cols="id_to_drop")
ApplyToCols(cols='id_to_drop', transformer=Cleaner(drop_if_unique=True))
>>> ApplyToCols(cleaner, cols="id_to_drop").fit_transform(df)
  text_to_keep
0          foo
1          bar
2          baz

For more advanced filtering operations, refer to the User Guide on
:ref:`user_guide_selectors` and the |ApplyToCols| documentation for details
on applying transformers to specific columns.
