.. _userguide_data_cleaning:

.. |DropUninformative| replace:: :class:`~skrub.DropUninformative`
.. |Cleaner| replace:: :class:`~skrub.Cleaner`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |deduplicate| replace:: :func:`~skrub.deduplicate`

Data Preparation with the ``skrub`` Transformers
------------------------------------------------

Cleaning dataframes and parsing datatypes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from skrub import Cleaner
    df_clean = Cleaner().fit_transform(df)

The |Cleaner| sanitizes dataframes by:

- Replacing strings that represent missing values with NA markers
- Dropping uninformative columns (add cross reference)
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

.. code-block:: python

    from skrub import Cleaner
    cleaner = Cleaner(numeric_dtype="float32")

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

Deduplicate categorical data with |deduplicate|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a series containing strings with typos, the |deduplicate| function
may be used to remove some typos by creating a mapping between the typo strings
and the correct strings. See the documentation for caveats and more detail.
