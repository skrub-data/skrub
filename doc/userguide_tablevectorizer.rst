... _userguide_tablevectorizer:

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
