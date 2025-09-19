
.. currentmodule:: skrub

.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |tabular_pipeline| replace:: :func:`~skrub.tabular_pipeline`
.. |HistGradientBoostingRegressor| replace:: :class:`~sklearn.ensemble.HistGradientBoostingRegressor`
.. |HistGradientBoostingClassifier| replace:: :class:`~sklearn.ensemble.HistGradientBoostingClassifier`
.. |Pipeline| replace:: :class:`~sklearn.pipeline.Pipeline`
.. |StandardScaler| replace:: :class:`~sklearn.preprocessing.StandardScaler`
.. |SimpleImputer| replace:: :class:`~sklearn.impute.SimpleImputer`

.. _user_guide_tabular_pipeline:

Building robust ML baselines with |tabular_pipeline|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The |tabular_pipeline| is a function that, given a scikit-learn estimator,
returns a full scikit-learn |Pipeline| that contains a |TableVectorizer|
followed by the given estimator.
If the estimator is a linear model (e.g., ``Ridge``, ``LogisticRegression``),
|tabular_pipeline| adds a |StandardScaler| and a |SimpleImputer| to the pipeline.

>>> from sklearn.linear_model import LinearRegression
>>> from skrub import tabular_pipeline
>>> tabular_pipeline(LinearRegression()) # doctest: +SKLEARN_VERSION >= "1.4" +ELLIPSIS
Pipeline(steps=[('tablevectorizer',
                 TableVectorizer(datetime=DatetimeEncoder(periodic_encoding='spline'))),
                ('simpleimputer', SimpleImputer(add_indicator=True)),
                ('standardscaler', StandardScaler()),
                ('linearregression', LinearRegression())])

It is also possible to call the function with the name of the task that must be
performed (``regression``/``regressor``, ``classification``/``classifier``) to
build a pipeline that uses a
|HistGradientBoostingRegressor|/|HistGradientBoostingClassifier|.

>>> from skrub import tabular_pipeline
>>> tabular_pipeline("regression") # doctest: +SKLEARN_VERSION >= "1.4" +ELLIPSIS
Pipeline(steps=[('tablevectorizer',
                 TableVectorizer(...),
                ('histgradientboostingregressor',
                 HistGradientBoostingRegressor())])

The pipeline prepared by |tabular_pipeline| is a strong first baseline for most
problems, but may not beat properly tuned ad-hoc pipelines.

.. list-table:: Parameter values choice of :class:`TableVectorizer` when using  the :func:`tabular_pipeline` function
   :header-rows: 1
   :widths: 25 25 25 25

   * - Parameter
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
   * - Numeric preprocessor
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
