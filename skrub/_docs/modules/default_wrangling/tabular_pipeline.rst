
.. currentmodule:: skrub

.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |tabular_pipeline| replace:: :func:`~skrub.tabular_pipeline`
.. |HistGradientBoostingRegressor| replace:: :class:`~sklearn.ensemble.HistGradientBoostingRegressor`
.. |HistGradientBoostingClassifier| replace:: :class:`~sklearn.ensemble.HistGradientBoostingClassifier`
.. |Pipeline| replace:: :class:`~sklearn.pipeline.Pipeline`
.. |SquashingScaler| replace:: :class:`~skrub.SquashingScaler`
.. |SimpleImputer| replace:: :class:`~sklearn.impute.SimpleImputer`
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
>>> tabular_pipeline(LinearRegression())
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
>>> tabular_pipeline("regression")
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
     - Native support
     - :class:`~sklearn.preprocessing.OneHotEncoder`
   * - High-cardinality encoder
     - :class:`StringEncoder`
     - :class:`StringEncoder`
     - :class:`StringEncoder`
   * - Numeric preprocessor
     - No processing
     - No processing
     - :class:`~skrub.SquashingScaler`
   * - Date preprocessor
     - :class:`DatetimeEncoder`
     - :class:`DatetimeEncoder`
     - :class:`DatetimeEncoder` with spline encoding
   * - Missing value strategy
     - Native support
     - Native support
     - :class:`~sklearn.impute.SimpleImputer`


The logic used by the tabular pipeline is quite simple
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The logic that is used by the |tabular_pipeline| is in fact quite simple, so
users do not lose much if they decide to write their own pipeline instead.
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

Extending the pipeline with the ``.steps`` attribute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the ``.steps`` attribute of the resulting pipeline together with
:func:`~sklearn.pipeline.make_pipeline` to build a new pipeline that has more
than just the steps in the tabular pipeline. The ``steps`` attribute of a
scikit-learn |Pipeline| is a list of ``(name, estimator)`` pairs, so we can
extract the estimators from the preprocessing steps and pass them to
``make_pipeline`` while inserting an extra transformation before the final
estimator:

>>> from sklearn.feature_selection import SelectPercentile, f_regression
>>> from sklearn.pipeline import make_pipeline
>>> from skrub import tabular_pipeline
>>> base_pipeline = tabular_pipeline("regressor")
>>> extended_pipeline = make_pipeline(
...     *[step[1] for step in base_pipeline.steps[:-1]],
...     SelectPercentile(score_func=f_regression, percentile=50),
...     base_pipeline.steps[-1][1],
... )
>>> [name for name, _ in extended_pipeline.steps]
['tablevectorizer', 'selectpercentile', 'histgradientboostingregressor']

Here ``[step[1] for step in base_pipeline.steps[:-1]]`` extracts the
estimators from all preprocessing steps, while omitting the final estimator.
Those preprocessing estimators are unpacked into ``make_pipeline``, then a
supervised feature-selection step and the original estimator are appended. This
pattern is useful whenever you want to add something such as feature selection,
dimensionality reduction, or calibration without rewriting the whole pipeline
from scratch.

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
>>> full_pipeline = tabular_pipeline(model_pipeline)
>>> [name for name, _ in full_pipeline.steps]
['tablevectorizer', 'simpleimputer', 'squashingscaler', 'pipeline']

The user-provided estimator pipeline is appended as a single final step. This
means that ``tabular_pipeline`` can still decide which preprocessing steps to
add before your own estimator logic.
