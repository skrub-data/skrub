.. _user_guide_part1:

User Guide Part 1: Transforming and Predicting with DataFrames
=============================================================

This guide showcases how to solve various problems using the features of ``skrub``. Short code snippets are provided to explain each operation. Additional examples are shown in the docstrings and in the ``examples`` section of the documentation.

`skrub` Objects for Transforming and Predicting with DataFrames
--------------------------------------------------------------

Exploring and Reporting DataFrames with the ``TableReport``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from skrub import TableReport
    TableReport(df)  # from a notebook cell
    TableReport(df).open()  # to open in a browser window

The table report gives a high-level overview of the given dataframe, suitable for quick exploratory analysis of series and dataframes. The report shows the first and last 5 rows of the dataframe (decided by the ``n_rows`` parameter), as well as additional information in other tabs.

- The **Stats** tab reports high-level statistics for each column.
- The **Distribution** tab collects summary plots for each column (max 30 by default).
- The **Associations** tab shows Cramer V and Pearson correlations between columns.
- Built-in filters allow selection of columns by dtype and other conditions.

In the **Distributions** tab, it is possible to select columns by clicking on the checkmark icon: the name of the column is added to the bar on top, so that it may be copied in a script.

Altering the Appearance of the ``TableReport``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For performance reasons, the ``TableReport`` disables the computation of Distributions and Associations for tables with more than 30 columns. This behavior can be changed by modifying the ``max_plot_columns`` and ``max_association_columns`` parameters, or by altering the configuration with ``set_config`` (refer to the ``TableReport`` and ``set_config`` docs for more detail).

More pre-computed examples are available at https://skrub-data.org/skrub-reports/examples/index.html. This doc example (add reference) showcases all the ``TableReport`` features on various datasets.

Exporting and Sharing the ``TableReport``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``TableReport`` is a standalone object that does not require a running notebook to be accessed after generation: it can be exported in HTML format and opened directly in a browser as a HTML page:

.. code-block:: python

    tr = TableReport(df)
    tr.write_html("report.html")  # save to file
    tr.html()  # get a string containing the HTML for a full page
    tr.html_snippet()  # get an HTML fragment to embed in a page
    tr.json()  # get the content of the report in JSON format

Finding Correlated Columns in a DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``TableReport``'s **Associations** tab shows this information. It is also possible to use the ``column_associations`` function, which returns a dataframe containing the associations.

Data Preparation with the ``skrub`` Transformers
------------------------------------------------

Cleaning DataFrames and Parsing Datatypes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from skrub import Cleaner
    df_clean = Cleaner().fit_transform(df)

The ``Cleaner`` sanitizes dataframes by:

- Replacing strings that represent missing values with NA markers
- Dropping uninformative columns (add cross reference)
- Parsing datetimes from datetime strings
- Forcing consistent categorical typing
- Converting columns to string, unless they have a more informative datatype (numerical, datetime, categorical)

Converting Numeric Dtypes to ``float32`` with the ``Cleaner``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the Cleaner parses numeric datatypes and does not cast them to a different dtype. In some cases, it may be beneficial to have the same numeric dtype for all numeric columns to guarantee compatibility between values.

The ``Cleaner`` allows conversion of numeric features to ``float32`` by setting the ``numeric_dtype`` parameter:

.. code-block:: python

    from skrub import Cleaner
    cleaner = Cleaner(numeric_dtype="float32")

Setting the dtype to ``float32`` reduces RAM footprint for most use cases and ensures that all missing values have the same representation. This also ensures compatibility with scikit-learn transformers.

Removing Unneeded Columns with ``DropUninformative`` and ``Cleaner``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tables may include columns that do not carry useful information. These columns increase computational cost and may reduce downstream performance.

The ``DropUninformative`` transformer includes various heuristics to drop columns considered "uninformative":

- Drops all columns that contain only missing values (threshold adjustable via ``drop_null_fraction``)
- Drops columns with only a single value if ``drop_if_constant=True``
- Drops string/categorical columns where each row is unique if ``drop_if_unique=True`` (use with care)

``DropUninformative`` is used by both ``TableVectorizer`` and ``Cleaner``; both accept the same parameters to drop columns accordingly.

Deduplicate Categorical Data with ``deduplicate``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a series containing strings with typos, the ``deduplicate`` function may be used to remove some typos by creating a mapping between the typo strings and the correct strings. See the documentation for caveats and more detail.

Handling Datetimes
------------------

Parsing Datetime Strings in DataFrames with ``ToDatetime`` and ``to_datetime``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``skrub`` includes objects to help with parsing and encoding datetimes.

- ``to_datetime`` and ``ToDatetime`` convert all columns in a dataframe that can be parsed as datetimes to the proper dtype.
- ``to_datetime`` is a function; ``ToDatetime`` is a scikit-learn compatible transformer.

.. code-block:: python

    from skrub import to_datetime, ToDatetime
    s = pd.Series(["2024-05-05T13:17:52", None, "2024-05-07T13:17:52"], name="when")
    to_datetime(s)
    ToDatetime().fit_transform(s)

Encoding and Feature Engineering on Datetimes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once datetimes have been parsed, they can be encoded as numerical features with the ``DatetimeEncoder``. This encoder extracts temporal features (year, month, day, hour, etc.) from datetime columns. No timezone conversion is done; the timezone in the feature is retained. The ``DatetimeEncoder`` rejects non-datetime columns, so it should only be applied after conversion using ``ToDatetime``.

Besides extracting datetime features, ``DatetimeEncoder`` can include additional time-based features, such as:

- Number of seconds from epoch (``add_total_seconds``)
- Day of the week (``add_weekday``)
- Day of the year (``add_day_of_year``)

Periodic encoding is supported through trigonometric (circular) and spline encoding: set the ``periodic_encoding`` parameter to ``circular`` or ``spline``.

Encoding String and Text Data as Numerical Features
--------------------------------------------------

In ``skrub``, categorical features are all features not detected as numeric or datetimes: this includes strings, text, IDs, and features with dtype ``categorical`` (e.g., ``pd.Categorical``).

High Cardinality and Low Cardinality Categorical Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In machine learning pipelines, these features are converted to numerical features using various encodings (``OneHotEncoding``, ``OrdinalEncoding``, etc.). Typically, categorical features are encoded using ``OneHotEncoding``, but this can cause issues when the number of unique values (the "cardinality") is very large.

The ``TableVectorizer`` classifies categorical features with more than 40 unique values as *high cardinality*, and all others as *low cardinality*. Different encoding strategies are applied to each kind; the threshold can be modified with the ``cardinality_threshold`` parameter.

- Low cardinality: encoded by default using scikit-learn ``OneHotEncoder``
- High cardinality: encoded using the ``StringEncoder``

Categorical encoding is applied only to columns that do not have a string or categorical dtype.

``StringEncoder``
~~~~~~~~~~~~~~~~~

A strong and quick baseline for both short strings with high cardinality and long text. Applies tf-idf vectorization followed by truncated SVD (Latent Semantic Analysis).

``TextEncoder``
~~~~~~~~~~~~~~~

Encodes string features using pretrained models from the HuggingFace Hub. It is a wrapper around ``SentenceTransformer`` compatible with the scikit-learn API and usable in pipelines. Best for free-flowing text and when columns include context found in the pretrained model.

``MinHashEncoder``
~~~~~~~~~~~~~~~~~~

Decomposes strings into ngrams, then applies the MinHash method to convert them into numerical features. Fast to train, but features may yield worse results compared to other methods.

``GapEncoder``
~~~~~~~~~~~~~~

Estimates "latent categories" on the training data, then encodes them as real numbers. Allows access to grouped features via ``.get_feature_names_out()``. May require a long time to train.

Comparison of the Categorical Encoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+------------------+---------------+-------------------------------+------------------------+--------------------------------------+
|     Encoder      | Training time | Performance on categorical     | Performance on text    | Notes                                |
|                  |               | data                          | data                   |                                      |
+==================+===============+===============================+========================+======================================+
| StringEncoder    | Fast          | Good                          | Good                   |                                      |
+------------------+---------------+-------------------------------+------------------------+--------------------------------------+
| TextEncoder      | Very slow     | Mediocre to good              | Very good              | Requires the ``transformers`` dep.   |
+------------------+---------------+-------------------------------+------------------------+--------------------------------------+
| GapEncoder       | Slow          | Good                          | Mediocre to good       | Interpretable                        |
+------------------+---------------+-------------------------------+------------------------+--------------------------------------+
| MinHashEncoder   | Very fast     | Mediocre to good              | Mediocre               |                                      |
+------------------+---------------+-------------------------------+------------------------+--------------------------------------+

Example 2 and this `blog post <https://en.wikipedia.org/wiki/Latent_semantic_analysis>`_ include a more systematic analysis of each method.

Building Strong Baselines with Robust Feature Engineering
--------------------------------------------------------

``TableVectorizer``
~~~~~~~~~~~~~~~~~~~

Performs feature engineering on dataframes by parsing the data type of each column and encoding columns according to their data type. Splits columns into four categories (high/low cardinality string, numerical, datetime) so each can be handled appropriately. High-cardinality: >40 unique values.

Numerical columns: left as is (``"passthrough"``)
Datetime columns: encoded with ``DatetimeEncoder``
High cardinality: uses ``StringEncoder``
Low cardinality: uses scikit-learn ``OneHotEncoder``

To change the encoder or alter default parameters, create a new encoder and pass it to ``TableVectorizer``.

.. code-block:: python

    from skrub import TableVectorizer, DatetimeEncoder, TextEncoder

    datetime_enc = DatetimeEncoder(periodic="circular")
    text_enc = TextEncoder()
    table_vec = TableVectorizer(datetime=datetime_enc, high_cardinality=text_enc)

``tabular_learner``
~~~~~~~~~~~~~~~~~~~

A function that, given a scikit-learn estimator or the name of the task (``regression``/``regressor``, ``classification``/``classifier``), returns a full scikit-learn compatible pipeline containing a ``TableVectorizer`` followed by the estimator, or a ``HistGradientBoostingRegressor``/``HistGradientBoostingClassifier``.

.. code-block:: python

    from skrub import tabular_learner
    from sklearn.linear_model import LinearRegression

    learner = tabular_learner("regression")
    learner = tabular_learner(LinearRegression())

If the estimator is a linear model (e.g., ``Ridge``, ``LogisticRegression``), ``tabular_learner`` adds a ``StandardScaler`` and a ``SimpleImputer`` to the pipeline. The pipeline prepared by ``tabular_learner`` is a strong first baseline for most problems, but may not beat properly tuned ad-hoc pipelines.

Joining Tables
--------------

``skrub`` features various objects that simplify combining information spread over multiple tables.

Approximate Join with ``fuzzy_join`` and ``Joiner``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``fuzzy_join`` function joins tables on approximate matches by vectorizing (embedding) the key columns in each table, then matching each row in the left table with its nearest neighbor in the right table according to the Euclidean distance between their embeddings.

The ``Joiner`` implements the same fuzzy join logic, but as a scikit-learn compatible transformer.

- Fuzzy joining with ``Joiner`` and ``fuzzy_join``
- ``AggJoiner`` and ``MultiAggJoiner``
- ``InterpolationJoiner``

`skrub` Utilities and Customization
-----------------------------------

Customizing ``skrub`` by Changing the Default Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``skrub`` includes a configuration manager that allows setting various parameters (see the ``set_config`` documentation for more detail).

It is possible to change configuration options using the ``set_config`` function:

.. code-block:: python

    from skrub import set_config
    set_config(use_tablereport=True)

Each configuration parameter can also be modified by setting its environment variable.

A ``config_context`` is also provided, which allows temporarily altering the configuration:

.. code-block:: python

    import skrub
    with skrub.config_context(max_plot_columns=1):
        ...

Datasets
--------

``skrub`` includes a number of datasets used for running examples. Each dataset can be downloaded using its ``fetch_*`` function, provided in the ``skrub.datasets`` namespace:

.. code-block:: python

    from skrub.datasets import fetch_employee_salaries
    data = fetch_employee_salaries()

Datasets are stored as ``Bunch`` objects, which include the full data, an ``X`` feature matrix, and a ``y`` target column with type ``pd.DataFrame``. Some datasets may have a different format depending on the use case.

Modifying the Download Location of ``skrub`` Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, datasets are stored in ``~/skrub_data``, where ``~`` is expanded as the (OS dependent) home directory of the user. The function ``get_data_dir`` shows the location that ``skrub`` uses to store data.

If needed, it is possible to change this location by modifying the environment variable ``SKRUB_DATA_DIRECTORY`` to an **absolute directory path**.
