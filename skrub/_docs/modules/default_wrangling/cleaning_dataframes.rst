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

The |Cleaner| aids with this by running common operations on each column, including
replacing "null-looking" strings (e.g., ``NULL``) with actual null values, and
parsing datetimes and numbers.

.. admonition:: All the transformations done by the |Cleaner|
  :collapsible: closed

  - Clean null strings: Replace strings typically used to represent missing values
    with a null value suitable for the column under consideration.

  - |DropUninformative|: Drop the column if it is considered "uninformative."
    A column is considered "uninformative" if it contains only missing values
    (``drop_null_fraction``), or only a constant value (``drop_if_constant``).
    By default, the |Cleaner| keeps all columns
    unless they contain only missing values. Refer to :ref:`user_guide_drop_uninformative`
    for more detail on this operation.

  - |ToDatetime|: Parse datetimes represented as strings and return them as
    actual datetimes with the correct dtype. If ``datetime_format`` is provided,
    it is forwarded to |ToDatetime|. Otherwise, the format is guessed according
    to common datetime formats.

  - Convert to strings: Convert columns to strings unless they have a more informative
    dtype, such as numeric, categorical, or datetime.

If ``parse_numbers`` is set to ``True``, the ``Cleaner`` will parse
string columns that contain only numbers and convert them to ``float32``.
If ``cast_to_float32=True``, the ``Cleaner`` will also convert numeric columns
(e.g. ``float64``, ``int64``) to ``float32``.

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
date    datetime64[...]
dtype:  ...

Note that the ``"all_missing"`` column has been dropped, and that the ``"date"``
column has been correctly parsed as a datetime column.

Parsing numeric-looking strings with the |Cleaner|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, when the |Cleaner| encounters a string series that contains only
numeric-looking values (for example ``["1", "2", "3"]``), it leaves it
unchanged.

The |Cleaner| can parse those values by setting ``parse_numbers=True``:

>>> from skrub import Cleaner
>>> cleaner = Cleaner(parse_numbers=True)
>>> import pandas as pd
>>> df = pd.DataFrame({
...     "id_as_str": ["1", "2", "3"],
...     "id": [1, 2, 3],
... })
>>> df.dtypes
id_as_str    ...
id           int64
dtype: ...
>>> df_cleaned = cleaner.fit_transform(df)
>>> df_cleaned.dtypes
id_as_str    float32
id           int64
dtype: ...

Parsed string values are converted to ``float32`` (not to ``int64`` or
``float64``), to keep a consistent numeric representation that is compatible
with downstream scikit-learn transformers.

When ``parse_numbers=False`` (default), both columns keep their original dtypes.

Downcasting float dtypes to ``float32`` with the |Cleaner|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, floating-point columns (e.g. ``float64``) keep their original dtype.
To downcast numeric columns to ``float32``, set
``cast_to_float32=True``:

>>> from skrub import Cleaner
>>> cleaner = Cleaner(cast_to_float32=True)
>>> import pandas as pd
>>> df = pd.DataFrame({
...     "f64": [1.0, 2.0, 3.0],
...     "i64": [1, 2, 3],
... })
>>> df.dtypes
f64    float64
i64      int64
dtype: ...
>>> cleaner.fit_transform(df).dtypes
f64    float32
i64    float32
dtype: ...
