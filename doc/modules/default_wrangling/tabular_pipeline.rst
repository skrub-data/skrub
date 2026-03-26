
.. currentmodule:: skrub

.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |tabular_pipeline| replace:: :func:`~skrub.tabular_pipeline`
.. |HistGradientBoostingRegressor| replace:: :class:`~sklearn.ensemble.HistGradientBoostingRegressor`
.. |HistGradientBoostingClassifier| replace:: :class:`~sklearn.ensemble.HistGradientBoostingClassifier`
.. |Pipeline| replace:: :class:`~sklearn.pipeline.Pipeline`
.. |SquashingScaler| replace:: :class:`~skrub.SquashingScaler`
.. |SimpleImputer| replace:: :class:`~sklearn.impute.SimpleImputer`
.. |ApplyToCols| replace:: :class:`~skrub.ApplyToCols`
.. |TextEncoder| replace:: :class:`~skrub.TextEncoder`
.. |ToCategorical| replace:: :class:`~skrub.ToCategorical`

.. _user_guide_tabular_pipeline:

Building robust ML baselines with |tabular_pipeline|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The |tabular_pipeline| is a function that, given a scikit-learn estimator,
returns a full scikit-learn |Pipeline| that contains a |TableVectorizer|
followed by the given estimator.
If the estimator is a linear model (e.g., ``Ridge``, ``LogisticRegression``),
|tabular_pipeline| adds a |SquashingScaler| and a |SimpleImputer| to the pipeline.

>>> from sklearn.linear_model import LinearRegression
>>> from skrub import tabular_pipeline
>>> tabular_pipeline(LinearRegression()) # doctest: +SKLEARN_VERSION >= "1.4" +ELLIPSIS
Pipeline(steps=[('tablevectorizer',
                 TableVectorizer(datetime=DatetimeEncoder(periodic_encoding='spline'))),
                ('simpleimputer', SimpleImputer(add_indicator=True)),
                ('squashingscaler', SquashingScaler(max_absolute_value=5)),
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
                 HistGradientBoostingRegressor(...))])

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
     - :class:`~sklearn.preprocessing.SquashingScaler`
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

Extending the pipeline with the ``.steps`` attribute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the ``.steps`` attribute of the resulting pipeline together with
:func:`~sklearn.pipeline.make_pipeline` to build a new pipeline that has more
than just the steps in the tabular pipeline. This is useful when you want to
keep skrub's default preprocessing, while inserting an extra transformation
before the final estimator:

>>> from sklearn.feature_selection import SelectPercentile, f_regression
>>> from sklearn.pipeline import make_pipeline
>>> from skrub import tabular_pipeline
>>> base_pipeline = tabular_pipeline("regressor")  # doctest: +SKLEARN_VERSION >= "1.4"
>>> extended_pipeline = make_pipeline(
...     *[step[1] for step in base_pipeline.steps[:-1]],
...     SelectPercentile(score_func=f_regression, percentile=50),
...     base_pipeline.steps[-1][1],
... )
>>> [name for name, _ in extended_pipeline.steps]  # doctest: +SKLEARN_VERSION >= "1.4"
['tablevectorizer', 'selectpercentile', 'histgradientboostingregressor']

Here we reused the preprocessing steps from ``base_pipeline``, inserted a
supervised feature-selection step, and kept the original estimator as the last
step. This pattern is useful whenever you want to add something such as feature
selection, dimensionality reduction, or calibration without rewriting the whole
pipeline from scratch.

Using a pipeline as the estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The estimator passed to |tabular_pipeline| can itself be a |Pipeline|. This is
often the simplest way to add estimator-specific postprocessing while keeping
the default table preprocessing:

>>> from sklearn.decomposition import PCA
>>> from sklearn.linear_model import Ridge
>>> from sklearn.pipeline import make_pipeline
>>> from skrub import tabular_pipeline
>>> model_pipeline = make_pipeline(PCA(n_components=20), Ridge())
>>> full_pipeline = tabular_pipeline(model_pipeline)  # doctest: +SKLEARN_VERSION >= "1.4"
>>> [name for name, _ in full_pipeline.steps]  # doctest: +SKLEARN_VERSION >= "1.4"
['tablevectorizer', 'simpleimputer', 'squashingscaler', 'pipeline']

The user-provided estimator pipeline is appended as a single final step. This
means that ``tabular_pipeline`` can still decide which preprocessing steps to
add before your own estimator logic.

Moving from the tabular pipeline to a custom-made pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also move gradually from |tabular_pipeline| to a more explicit pipeline.
Because the result is an ordinary scikit-learn pipeline, you can take some of
its steps, change the parameters that matter for your data, and assemble a new
pipeline yourself.

For example, you can start from the default regressor pipeline, then replace the
high-cardinality encoder in the |TableVectorizer| with a |TextEncoder|:

>>> from sklearn.base import clone
>>> from skrub import TextEncoder, tabular_pipeline
>>> base = tabular_pipeline("regressor")  # doctest: +SKLEARN_VERSION >= "1.4"
>>> vectorizer = clone(base.named_steps["tablevectorizer"]).set_params(
...     high_cardinality=TextEncoder()
... )  # doctest: +SKLEARN_VERSION >= "1.4"
>>> vectorizer  # doctest: +ELLIPSIS +SKLEARN_VERSION >= "1.4"
TableVectorizer(high_cardinality=TextEncoder(), ...)

This pattern generalizes to any parameter of the vectorizer, and also to the
other steps returned by |tabular_pipeline|.

You can also combine |ApplyToCols| and the |TableVectorizer| to transform only
part of the table before applying the vectorizer. For example, you may want to
cast an identifier column to a categorical dtype, while leaving the rest of the
automatic logic unchanged:

>>> import pandas as pd
>>> from sklearn.linear_model import Ridge
>>> from sklearn.pipeline import make_pipeline
>>> from skrub import ApplyToCols, TableVectorizer, ToCategorical
>>> df = pd.DataFrame({"id": ["1", "2", "3"], "value": [10, 20, 30]})
>>> custom_pipeline = make_pipeline(
...     ApplyToCols(ToCategorical(), cols=["id"]),
...     TableVectorizer(),
...     Ridge(),
... )
>>> [name for name, _ in custom_pipeline.steps]
['applytocols', 'tablevectorizer', 'ridge']

This kind of composition is often enough when only a few columns need special
treatment and the rest of the dataframe can still go through the default
|TableVectorizer| behavior.

The logic used by the tabular pipeline is quite simple
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The logic that is used by the |tabular_pipeline| is in fact quite simple, so
users do not miss on a lot if they decide to write their own pipeline instead.
In practice it does only three things:

- It chooses a |TableVectorizer| configuration from the estimator type. For
  example, linear models get spline datetime features, while histogram gradient
  boosting models with ``categorical_features="from_dtype"`` get
  ``low_cardinality=ToCategorical()``.
- It inserts a |SimpleImputer| when the estimator cannot handle missing values.
- It inserts a |SquashingScaler| for estimators that benefit from scaling, and
  skips it for tree ensembles.

If your use case needs more control, writing the full pipeline yourself is
usually straightforward and gives you access to the exact same building blocks.
See the source of :func:`~skrub.tabular_pipeline` for the exact logic.
