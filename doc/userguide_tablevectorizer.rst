.. _userguide_tablevectorizer:
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |DatetimeEncoder| replace:: :class:`~skrub.DatetimeEncoder`
.. |StringEncoder| replace:: :class:`~skrub.StringEncoder`
.. |OneHotEncoder| replace:: :class:`~sklearn.preprocessing.OneHotEncoder`
.. |TextEncoder| replace:: :class:`~skrub.TextEncoder`
.. |tabular_learner| replace:: :func:`~skrub.tabular_learner`
.. |HistGradientBoostingRegressor| replace:: :class:`~sklearn.ensemble.HistGradientBoostingRegressor`
.. |HistGradientBoostingClassifier| replace:: :class:`~sklearn.ensemble.HistGradientBoostingClassifier`

Building Strong Baselines with Robust Feature Engineering
--------------------------------------------------------

|TableVectorizer|
~~~~~~~~~~~~~~~~~

The |TableVectorizer| performs feature engineering on dataframes by parsing the
data type of each column and encoding columns according to their data type,
producing new numeric features that can be used in machine learning models.

The |TableVectorizer| splits columns into four categories:

- High-cardinality categorical columns: >40 unique values
- Low-cardinality categorical columns: ≤40 unique values
- Numerical columns
- Datetime columns

Then, the default encoders for each category are applied:

- Numerical columns: left as is (``"passthrough"``)
- Datetime columns: encoded with |DatetimeEncoder|
- High cardinality: uses |StringEncoder|
- Low cardinality: uses scikit-learn |OneHotEncoder|

To change the encoder or alter default parameters, create a new encoder and pass
it to |TableVectorizer|.

.. code-block:: python

    from skrub import TableVectorizer, DatetimeEncoder, TextEncoder

    datetime_enc = DatetimeEncoder(periodic="circular")
    text_enc = TextEncoder()
    table_vec = TableVectorizer(datetime=datetime_enc, high_cardinality=text_enc)

|tabular_learner|
~~~~~~~~~~~~~~~~~~
The |tabular_learner| is a function that, given a scikit-learn estimator or the
 name of the task (``regression``/``regressor``, ``classification``/``classifier``),
 returns a full scikit-learn pipeline that contains a |TableVectorizer|
 followed by the given estimator, or a
 |HistGradientBoostingRegressor|/|HistGradientBoostingClassifier| if only
 the name of the task is given.

.. code-block:: python

    from skrub import tabular_learner
    from sklearn.linear_model import LinearRegression

    learner = tabular_learner("regression")
    learner = tabular_learner(LinearRegression())

If the estimator is a linear model (e.g., ``Ridge``, ``LogisticRegression``),
|tabular_learner| adds a ``StandardScaler`` and a ``SimpleImputer`` to the pipeline.
The pipeline prepared by |tabular_learner| is a strong first baseline for most
problems, but may not beat properly tuned ad-hoc pipelines.
