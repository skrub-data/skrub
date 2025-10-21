.. |DropUninformative| replace:: :class:`~skrub.DropUninformative`
.. |Cleaner| replace:: :class:`~skrub.Cleaner`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |ToDatetime| replace:: :class:`~skrub.ToDatetime`

.. _user_guide_cleaning_dataframes:

|Cleaner|: sanitizing a dataframe
---------------------------------

Very often, the first steps in preparing a dataframe for further use involve
understanding the datatypes in the data and changing them into a more suitable format
(e.g., from string to number or datetime).

The |Cleaner| aids with this by running the following set of transformations on
each column:

- Clean null strings: Replace strings typically used to represent missing values
  with a null value suitable for the column under consideration.

- |DropUninformative|: Drop the column if it is considered "uninformative."
  A column is considered "uninformative" if it contains only missing values
  (``drop_null_fraction``), only a constant value (``drop_if_constant``), or if all
  values are distinct (``drop_if_unique``). By default, the |Cleaner| keeps all columns
  unless they contain only missing values. Refer to :ref:`user_guide_drop_uninformative`
  for more detail on this operation.

.. note::

  Setting ``drop_if_unique`` to ``True`` may lead to dropping columns
  that contain text or IDs. Numeric columns are never dropped by ``drop_if_unique``.

- |ToDatetime|: Parse datetimes represented as strings and return them as
  actual datetimes with the correct dtype. If ``datetime_format`` is provided,
  it is forwarded to |ToDatetime|. Otherwise, the format is guessed according
  to common datetime formats.

- Clean categories: If the dtype of the column is detected as "Categorical",
  process it based on the dataframe library (Pandas or Polars) to ensure
  consistent typing and avoid downstream issues.

- Convert to strings: Convert columns to strings unless they have a more informative
  dtype, such as numeric, categorical, or datetime.

If ``numeric_dtype`` is set to ``float32``, the ``Cleaner`` will also convert
numeric columns to ``np.float32`` dtype, ensuring a consistent representation
of numbers and missing values. This can be useful if the ``Cleaner``
is used as a preprocessing step at the beginning of an ML pipeline.

The |Cleaner| is a scikit-learn compatible transformer:

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
>>> df_clean.dtypes
id               int64
date    datetime64[ns]
dtype: object

Note that the ``"all_missing"`` column has been dropped, and that the ``"date"``
column has been correctly parsed as a datetime column.

Converting numeric dtypes to ``float32`` with the |Cleaner|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, when the |Cleaner| encounters numeric dtypes (e.g., ``int8``,
``float64``), it leaves them as-is. In some cases, it may be beneficial to have
the same numeric dtype for all numeric columns to guarantee compatibility between
values.

The |Cleaner| allows conversion of numeric features to ``float32`` by setting
the ``numeric_dtype`` parameter:

>>> from skrub import Cleaner
>>> cleaner = Cleaner(numeric_dtype="float32")
>>> import pandas as pd
>>> df = pd.DataFrame({
...     "id": [1, 2, 3],
... })
>>> df.dtypes
id    int64
dtype: object
>>> df_cleaned = cleaner.fit_transform(df)
>>> df_cleaned.dtypes
id    float32
dtype: object

Setting the dtype to ``float32`` reduces RAM footprint for most use cases and
ensures that all missing values have the same representation. This also ensures
compatibility with scikit-learn transformers.
