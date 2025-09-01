.. |DropUninformative| replace:: :class:`~skrub.DropUninformative`
.. |Cleaner| replace:: :class:`~skrub.Cleaner`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |deduplicate| replace:: :func:`~skrub.deduplicate`
.. |SquashingScaler| replace:: :class:`~skrub.SquashingScaler`
.. |RobustScaler| replace:: :class:`~sklearn.preprocessing.RobustScaler`

.. _userguide_data_cleaning:

Data Preparation with ``skrub`` Transformers
---------------------------------------------

Cleaning dataframes and parsing datatypes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> from skrub import Cleaner
>>> import pandas as pd
>>> df = pd.DataFrame({
...     "id": [1, 2, 3],
...     "all_missing": ["", "", ""],
...     "date": ["2024-05-05", "2024-05-06", "2024-05-07"],
... })
>>> df_clean = Cleaner().fit_transform(df)
>>> df_clean
      id       date
  0   1 2024-05-05
  1   2 2024-05-06
  2   3 2024-05-07

The |Cleaner| converts data types and Nan values in dataframes to ease downstream preprocessing. It includes:

- Replacing strings that represent missing values with NA markers
- Dropping uninformative columns (see |DropUninformative|)
- Parsing datetimes from datetime strings
- Forcing consistent categorical typing
- Converting columns to string, unless they have a more informative datatype (numerical, datetime, categorical)

Converting numeric dtypes to ``float32`` with the |Cleaner|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the |Cleaner| parses numeric datatypes and does not cast them to a
different dtype. In some cases, it may be beneficial to have the same numeric
dtype for all numeric columns to guarantee compatibility between values.

The |Cleaner| allows conversion of numeric features to ``float32`` by setting
the ``numeric_dtype`` parameter:

>>> from skrub import Cleaner
>>> cleaner = Cleaner(numeric_dtype="float32")

Setting the dtype to ``float32`` reduces RAM footprint for most use cases and
ensures that all missing values have the same representation. This also ensures
compatibility with scikit-learn transformers.

Removing unneeded columns with |DropUninformative| and |Cleaner|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|DropUninformative| is used to remove features or data points that do not provide
useful information for the analysis or model.

Tables may include columns that do not carry useful information. These columns
increase computational cost and may reduce downstream performance.

The |DropUninformative| transformer includes various heuristics to drop columns
considered "uninformative":

- Drops all columns that contain only missing values (threshold adjustable via
  ``drop_null_fraction``)
- Drops columns with only a single value if ``drop_if_constant=True``
- Drops string/categorical columns where each row is unique if
  ``drop_if_unique=True`` (use with care)

|DropUninformative| is used by both |TableVectorizer| and |Cleaner|; both accept
the same parameters to drop columns accordingly.

Robust scaling of numerical features using |SquashingScaler|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The |SquashingScaler| is a robust scaler for numerical features, particularly
useful when features include outliers (such as infinite values); missing values
are left unchanged (they are not interpolated).
The |SquashingScaler| centers and scales the data in such a way that outliers are
less likely to skew the final result compared to alternative methods.

Based on the specified ``quantile_range`` parameter, the scaler employs a scikit-learn
|RobustScaler| to rescale the values in a way that the quantile range occupies
interval of length two, centering the median to zero. It therefore ensures that
inliers are spread to a reasonable range. Afterwards, it uses a smooth clipping
function to ensure all values (including outliers and infinite values) are in the
range ``[-max_absolute_value, max_absolute_value]``.

More information about the theory behind the scaler is available in the
|SquashingScaler| documentation, while this
:ref:`working example <sphx_glr_auto_examples_11_squashing_scaler.py>` compares
different scalers when used on data that include outliers.

Deduplicate categorical data with |deduplicate|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a series containing strings with typos, the |deduplicate| function
may be used to remove some typos by creating a mapping between the typo strings
and the correct strings. See the documentation for caveats and more detail.
