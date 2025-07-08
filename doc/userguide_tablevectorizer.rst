.. _userguide_tablevectorizer:
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |DatetimeEncoder| replace:: :class:`~skrub.DatetimeEncoder`
.. |StringEncoder| replace:: :class:`~skrub.StringEncoder`
.. |OneHotEncoder| replace:: :class:`~sklearn.preprocessing.OneHotEncoder`
.. |TextEncoder| replace:: :class:`~skrub.TextEncoder`
.. |tabular_learner| replace:: :func:`~skrub.tabular_learner`
.. |HistGradientBoostingRegressor| replace:: :class:`~sklearn.ensemble.HistGradientBoostingRegressor`
.. |HistGradientBoostingClassifier| replace:: :class:`~sklearn.ensemble.HistGradientBoostingClassifier`
.. |StandardScaler| replace:: :class:`~sklearn.preprocessing.StandardScaler`
.. |SimpleImputer| replace:: :class:`~sklearn.impute.SimpleImputer`

Strong baseline pipelines
--------------------------------------------------------

|TableVectorizer|
~~~~~~~~~~~~~~~~~
In tabular machine learning pipelines, categorical features are converted to numerical features
using various encodings (|OneHotEncoder|, |OrdinalEncoder|, etc.).

The |TableVectorizer| performs feature engineering on dataframes by parsing the
data type of each column and encoding columns according to their data type,
producing new numeric features that can be used in machine learning models.

The |TableVectorizer| splits columns into four categories and applies a default
encoder to each.

+-------------------------------+---------------------------------------------+
| Column category               | Default encoder                             |
+===============================+=============================================+
| High-cardinality categorical  | |StringEncoder|                             |
| columns.                      |                                             |
+-------------------------------+---------------------------------------------+
| Low-cardinality categorical   | scikit-learn |OneHotEncoder|                |
| columns                       |                                             |
+-------------------------------+---------------------------------------------+
| Numerical columns             | "passthrough" (no transformation)           |
+-------------------------------+---------------------------------------------+
| Datetime columns              | |DatetimeEncoder|                           |
+-------------------------------+---------------------------------------------+

**High cardinality** categorical columns are those with more than 40 unique values,
while all other categorical columns are considered **low cardinality**: the
threshold can be changed by setting the ``cardinality_threshold`` parameter of
|TableVectorizer|.

To change the encoder or alter default parameters, instantiate an encoder and pass
it to |TableVectorizer|.

.. code-block:: python

    from skrub import TableVectorizer, DatetimeEncoder, TextEncoder

    datetime_enc = DatetimeEncoder(periodic="circular")
    text_enc = TextEncoder()
    table_vec = TableVectorizer(datetime=datetime_enc, high_cardinality=text_enc)

The |TableVectorizer| is used in :ref:`example_string_encodings`, while the
docstring of the class provides more details on the parameters and usage, as well
as various examples.



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
|tabular_learner| adds a |StandardScaler| and a |SimpleImputer| to the pipeline.
The pipeline prepared by |tabular_learner| is a strong first baseline for most
problems, but may not beat properly tuned ad-hoc pipelines.
