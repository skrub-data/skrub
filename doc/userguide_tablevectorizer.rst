.. _userguide_tablevectorizer:
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |DatetimeEncoder| replace:: :class:`~skrub.DatetimeEncoder`
.. |StringEncoder| replace:: :class:`~skrub.StringEncoder`
.. |OneHotEncoder| replace:: :class:`~sklearn.preprocessing.OneHotEncoder`
.. |OrdinalEncoder| replace:: :class:`~sklearn.preprocessing.OrdinalEncoder`
.. |TextEncoder| replace:: :class:`~skrub.TextEncoder`
.. |tabular_pipeline| replace:: :func:`~skrub.tabular_pipeline`
.. |HistGradientBoostingRegressor| replace:: :class:`~sklearn.ensemble.HistGradientBoostingRegressor`
.. |HistGradientBoostingClassifier| replace:: :class:`~sklearn.ensemble.HistGradientBoostingClassifier`
.. |StandardScaler| replace:: :class:`~sklearn.preprocessing.StandardScaler`
.. |SimpleImputer| replace:: :class:`~sklearn.impute.SimpleImputer`

Strong baseline pipelines
--------------------------------------------------------

|TableVectorizer|
~~~~~~~~~~~~~~~~~
In tabular machine learning pipelines, practitioners often convert categorical features to numerical features
using various encodings (|OneHotEncoder|, |OrdinalEncoder|, etc.).

The |TableVectorizer| parses the data type of each column and maps each column to an encoder, in order
to produce numeric features for machine learning models.

More precisely, the |TableVectorizer| maps columns to one of the following four groups by default:

- **High-cardinality categorical columns**: |StringEncoder|
- **Low-cardinality categorical columns**: scikit-learn |OneHotEncoder|
- **Numerical columns**: "passthrough" (no transformation)
- **Datetime columns**: |DatetimeEncoder|

**High cardinality** categorical columns are those with more than 40 unique values,
while all other categorical columns are considered **low cardinality**: the
threshold can be changed by setting the ``cardinality_threshold`` parameter of
|TableVectorizer|.

To change the encoder or alter default parameters, instantiate an encoder and pass
it to |TableVectorizer|.

>>> from skrub import TableVectorizer, DatetimeEncoder, TextEncoder

>>> datetime_enc = DatetimeEncoder(periodic_encoding="circular")
>>> text_enc = TextEncoder()
>>> table_vec = TableVectorizer(datetime=datetime_enc, high_cardinality=text_enc)

The |TableVectorizer| is used in :ref:`example_encodings`, while the
docstring of the class provides more details on the parameters and usage, as well
as various examples.



|tabular_pipeline|
~~~~~~~~~~~~~~~~~~
The |tabular_pipeline| is a function that, given a scikit-learn estimator or the
name of the task (``regression``/``regressor``, ``classification``/``classifier``),
returns a full scikit-learn pipeline that contains a |TableVectorizer|
followed by the given estimator, or a
|HistGradientBoostingRegressor|/|HistGradientBoostingClassifier| if only
the name of the task is given.


>>> from skrub import tabular_pipeline
>>> from sklearn.linear_model import LinearRegression

>>> learner = tabular_pipeline("regression")
>>> learner = tabular_pipeline(LinearRegression())

If the estimator is a linear model (e.g., ``Ridge``, ``LogisticRegression``),
|tabular_pipeline| adds a |StandardScaler| and a |SimpleImputer| to the pipeline.
The pipeline prepared by |tabular_pipeline| is a strong first baseline for most
problems, but may not beat properly tuned ad-hoc pipelines.

.. list-table:: Parameter values choice of :class:`TableVectorizer` when using the :func:`tabular_pipeline` function
   :header-rows: 1

   * -
     - ``RandomForest`` models
     - ``HistGradientBoosting`` models
     - Linear models and others
   * - Low-cardinality encoder
     - :class:`~sklearn.preprocessing.OrdinalEncoder`
     - Native support :sup:`(1)`
     - :class:`~sklearn.preprocessing.OneHotEncoder`
   * - High-cardinality encoder
     - :class:`StringEncoder`
     - :class:`StringEncoder`
     - :class:`StringEncoder`
   * - Numerical preprocessor
     - No processing
     - No processing
     - :class:`~sklearn.preprocessing.StandardScaler`
   * - Date preprocessor
     - :class:`DatetimeEncoder`
     - :class:`DatetimeEncoder`
     - :class:`DatetimeEncoder` with spline encoding
   * - Missing value strategy
     - Native support :sup:`(2)`
     - Native support
     - :class:`~sklearn.impute.SimpleImputer`

.. note::
  :sup:`(1)` if scikit-learn installed is lower than 1.4, then
  :class:`~sklearn.preprocessing.OrdinalEncoder` is used since native support
  for categorical features is not available.

  :sup:`(2)` if scikit-learn installed is lower than 1.4, then
  :class:`~sklearn.impute.SimpleImputer` is used since native support
  for missing values is not available.
