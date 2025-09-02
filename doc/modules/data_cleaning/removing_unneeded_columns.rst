.. _removing_unneeded_columns:

.. |DropUninformative| replace:: :class:`~skrub.DropUninformative`
.. |ApplyToCols| replace:: :class:`~skrub.ApplyToCols`
.. |Cleaner| replace:: :class:`~skrub.Cleaner`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`

Removing unneeded columns with |DropUninformative| and |Cleaner|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tables may include columns that do not carry useful information. These columns
increase computational cost and may reduce downstream performance.

The |DropUninformative| transformer is used to remove features that do not
provide useful information for the analysis or model, by using various heuristics
to select columns that may be considered "uninformative", and thus potentially
problematic.

The heuristics used include:

- Dropping all columns that contain a fraction of missing values larger than the given
  threshold. By default, the threshold is 1, so that only columns that contain
  only missing values are dropped. The threshold can be adjusted by setting the
  ``drop_null_fraction`` parameter to be lower than 1. Setting ``drop_null_fraction``
  to ``None`` disables this check and keeps all columns.
- Dropping constant columns, that is columns that contain only a single value. This
  check is controlled by the ``drop_if_constant`` parameter, and is set to ``False``
  by default. Note that missing values are considered as an additional distinct value
  by this heuristic, so a constant column that contains missing values *will not
  be dropped*.
- Dropping string/categorical columns where each row has a unique value. This may
  be useful if a column is an alphanumeric ID that does not bring information.
  By default, the relative parameter ``drop_if_unique`` is set to ``False``. Note
  that setting this parameter to ``True`` may lead to dropping columns that contain
  free-flowing text. Additionally, this check is done only on string/categorical
  columns to avoid dropping numerical columns by mistake.

|DropUninformative| is used by both |TableVectorizer| and |Cleaner|; both accept
the same parameters to drop columns accordingly.


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


We want to drop constant columns, and columns that contain only single values.

>>> cleaner = Cleaner(drop_if_constant=True)
>>> df_cleaned = cleaner.fit_transform(df)
>>> df_cleaned
   B  D
0  2  4
1  3  5
2  2  6

Applying |DropUninformative| only to a subset of columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to combine the |DropUninformative| transformer to a subset of columns
using |ApplyToCols|.

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

Dropping unique columns in this dataframe returns an empty dataframe:
>>> cleaner = Cleaner(drop_if_unique=True)
>>> cleaner.fit_transform(df)
Empty DataFrame
Columns: []
Index: [0, 1, 2]

In order to apply the transformer only on the ``id_to_drop`` column, we can use
|ApplyToCols|:

>>> ApplyToCols(cleaner, cols="id_to_drop")
ApplyToCols(cols='id_to_drop', transformer=Cleaner(drop_if_unique=True))
>>> ApplyToCols(cleaner, cols="id_to_drop").fit_transform(df)
  text_to_keep
0          foo
1          bar
2          baz

It is possible to apply complex filtering operations in order to apply transformers
only to specific subsets of columns:
refer to User Guide on :ref:`user_guide_selectors`  and the documentation of
|ApplyToCols| for more detail on how to apply transformers to specific columns.
