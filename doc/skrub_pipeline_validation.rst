.. currentmodule:: skrub

.. _skrub_pipeline_validation:

=====================================
Tuning and validating Skrub Pipelines
=====================================

To evaluate the prediction performance of our pipeline, we can fit it on a training dataset, then obtaining prediction
on an unseen, test dataset.

In scikit-learn, we pass to estimators and pipelines an ``X`` and ``y`` matrix
with one row per observation from the start. Therefore, we can split the
data into a training and test set independently from the pipeline.

However, in many real-world scenarios, our data sources are not already
organized into ``X`` and ``y`` matrices. Some transformations may be necessary to
build them, and we want to keep those transformations inside the pipeline so
that they can be reliably re-applied to new data.

Therefore, we must start our pipeline by creating the design matrix and targets,
then tell skrub which intermediate results in the pipeline constitute ``X`` and
``y`` respectively.

Let us consider a toy example where we simply obtain ``X`` and
``y`` from a single table. More complex transformations would be handled in
the same way.

>>> from sklearn.datasets import load_diabetes
>>> from sklearn.linear_model import Ridge
>>> import skrub

>>> diabetes_df = load_diabetes(as_frame=True)["frame"]

In the original data, all features and the target are in the same dataframe.

>>> data = skrub.var("data", diabetes_df)

We build our design matrix by dropping the target. Note we use
``errors="ignore"`` so that pandas does not raise an error if the column we want
to drop is already missing. Indeed, when we will need to make actual useful
predictions on unlabelled data, the "target" column will not be available.

>>> X = data.drop(columns="target", errors="ignore").skb.mark_as_X()

We use :meth:`.skb.mark_as_X() <Expr.skb.mark_as_X>` to indicate that this
intermediate result (the dataframe obtained after dropping "target") is the
``X`` design matrix. This is the dataframe that will be split into a training
and a testing part when we split our dataset or perform cross-validation.

Similarly for ``y``, we use :meth:`.skb.mark_as_y() <Expr.skb.mark_as_y>`:

>>> y = data["target"].skb.mark_as_y()

Now we can add our supervised estimator:

>>> pred = X.skb.apply(Ridge(), y=y)
>>> pred
<Apply Ridge>
Result:
―――――――
         target
0    182.673354
1     90.998607
2    166.113476
3    156.034880
4    133.659575
..          ...
437  180.323365
438  135.798908
439  139.855630
440  182.645829
441   83.564413
[442 rows x 1 columns]


Once a pipeline is defined and the ``X`` and ``y`` nodes are identified, skrub
is able to split the dataset and perform cross-validation.

Splitting
---------

We can use :meth:`.skb.train_test_split() <Expr.skb.train_test_split>` to
perform a single train-test split. Skrub first evaluates the expressions on
which we used :meth:`.skb.mark_as_X() <Expr.skb.mark_as_X>` and
:meth:`.skb.mark_as_y() <Expr.skb.mark_as_y>`: the first few steps of the
pipeline are executed until we have a value for ``X`` and for ``y``. Then, those
dataframes are split using the provided splitter function (by default
scikit-learn's :func:`sklearn.model_selection.train_test_split`).

>>> split = pred.skb.train_test_split(shuffle=False)
>>> split.keys()
dict_keys(['train', 'test', 'X_train', 'X_test', 'y_train', 'y_test'])

``train`` and ``test`` are the full dictionaries corresponding to the training
and testing data. The corresponding ``X`` and ``y`` are the values, in those
dictionaries, for the nodes marked with
:meth:`.skb.mark_as_X() <Expr.skb.mark_as_X>`
and :meth:`.skb.mark_as_y() <Expr.skb.mark_as_y>`.

We can now fit our pipeline on the training data:

>>> pipeline = pred.skb.get_pipeline()
>>> pipeline.fit(split["train"])
SkrubPipeline(expr=<Apply Ridge>)

Only the training part of ``X`` and ``y`` are used. The subsequent steps are
evaluated, using this data, to fit the rest of the pipeline.

And we can obtain predictions on the test part:

>>> test_pred = pipeline.predict(split["test"])
>>> test_y_true = split["y_test"]

>>> from sklearn.metrics import r2_score

>>> r2_score(test_y_true, test_pred)
0.440999149220359

Cross-validation
----------------

We can increase our confidence in our score by using cross-validation instead of
a single split. The same mechanism is used but we now fit and evaluate the model
on several splits. This is done with :meth:`.skb.cross_validate()
<Expr.skb.cross_validate>`.

>>> pred.skb.cross_validate() # doctest: +SKIP
   fit_time  score_time  test_score
0  0.002816    0.001344    0.321665
1  0.002685    0.001323    0.440485
2  0.002468    0.001308    0.422104
3  0.002748    0.001321    0.424661
4  0.002649    0.001309    0.441961

Tuning choices
--------------

Skrub provides a convenient way to declare ranges of possible values, and tune
those choices to keep the values that give the best predictions on a validation
set.

Rather than specifying a grid of hyperparameters separately from the pipeline,
we simply insert special skrub objects in place of the value. For example we
replace the hyperparameter ``alpha`` (which should be a float) with a range
created by :func:`skrub.choose_float`. Skrub can use it to select the best value
for ``alpha``.

>>> pred = X.skb.apply(
...     Ridge(alpha=skrub.choose_float(0.01, 10.0, log=True, name="α")), y=y
... )

.. warning::

   When we do :meth:`.skb.get_pipeline() <Expr.skb.get_pipeline>`, the pipeline
   we obtain does not perform any hyperparameter tuning. The pipeline we obtain
   uses default values for each of the choices. For numeric choices it is the
   middle of the range, and for :func:`choose_from` it is the first option we
   give it.

   To get a pipeline that runs an internal cross-validation to select the best
   hyperparameters, we must use :meth:`.skb.get_grid_search()
   <Expr.skb.get_grid_search()>` or :meth:`.skb.get_randomized_search()
   <Expr.skb.get_randomized_search>`.


Here are the different kinds of choices, along with their default outcome when
we are not using hyperparameter search:

.. _choice-defaults-table:

.. list-table:: default choice outcomes
   :header-rows: 1

   * - choosing function
     - description
     - default outcome
   * - :func:`choose_from([10, 20]) <choose_from>`
     - choose between the listed options 10 and 20
     - first outcome in the list: ``10``
   * - :func:`choose_from({"a_name": 10, "b_name": 20}) <choose_from>`
     - choose between the listed options 10 and 20, dictionary keys are names
       for the options.
     - first outcome in the dict: ``10``
   * - :func:`optional(10) <optional>`
     - choose between the provided value and ``None`` (useful for optional
       transformations in a pipeline eg ``optional(StandardScaler())``).
     - the provided ``value``: ``10``
   * - :func:`choose_bool() <choose_bool>`
     - choose between True and False.
     - ``True``
   * - :func:`choose_float(1.0, 100.0) <choose_float>`
     - sample a floating-point number in a range.
     - the middle of the range: ``50.5``
   * - :func:`choose_int(1, 100) <choose_int>`
     - sample an integer in a range.
     - the int closest to the middle of the range: ``50``
   * - :func:`choose_float(1.0, 100.0, log=True) <choose_float>`
     - sample a float in a range on a logarithmic scale.
     - the middle of the range on a log scale: ``10.0``
   * - :func:`choose_int(1, 100, log=True) <choose_int>`
     - sample an int in a range on a logarithmic scale.
     - the int closest to the middle of the range on a log scale: ``10``
   * - :func:`choose_float(1.0, 100.0, n_steps=4) <choose_float>`
     - sample a float on a grid.
     - the step closest to the middle of the range: ``34.0`` (here steps are
       ``[1.0, 34.0, 67.0, 100.0]``)
   * - :func:`choose_int(1, 100, n_steps=4) <choose_int>`
     - sample an int on a grid.
     - the (integer) step closest to the middle of the range: ``34`` (here steps are
       ``[1, 34, 67, 100]``)
   * - :func:`choose_float(1.0, 100.0, log=True, n_steps=4) <choose_float>`
     - sample a float on a logarithmically-spaced grid.
     - the step closest to the middle of the range on a log scale: ``4.64``
       (here steps are ``[1.0, 4.64, 21.54, 100.0]``)
   * - :func:`choose_float(1.0, 100.0, log=True) <choose_int>`
     - sample an int on a logarithmically-spaced grid.
     - the (integer) step closest to the middle of the range on a log scale: ``5``
       (here steps are ``[1, 5, 22, 100]``)


The default choices for an expression, those that get used when calling
:meth:`.skb.get_pipeline() <Expr.skb.get_pipeline>`, can be inspected with
:meth:`.skb.describe_defaults() <Expr.skb.describe_defaults>`:

>>> pred.skb.describe_defaults()
{'α': 0.316...}

We can then find the best hyperparameters.

>>> search = pred.skb.get_randomized_search(fitted=True)
>>> search.results_  # doctest: +SKIP
   mean_test_score         α
0         0.478338  0.141359
1         0.476022  0.186623
2         0.474905  0.205476
3         0.457807  0.431171
4         0.456808  0.443038
5         0.439670  0.643117
6         0.420917  0.866328
7         0.380719  1.398196
8         0.233172  4.734989
9         0.168444  7.780156

Rather than fitting a randomized or grid search to find the best combination, it is also
possible to obtain an iterator over different parameter combinations, to inspect
their outputs or to have manual control over the model selection, using
:meth:`.skb.iter_pipelines_grid() <Expr.skb.iter_pipelines_grid>` or
:meth:`.skb.iter_pipelines_randomized() <Expr.skb.iter_pipelines_randomized>`.
Those yield the candidate pipelines that are explored by the grid and randomized
search respectively.

A human-readable description of parameters for a pipeline can be obtained with
:meth:`SkrubPipeline.describe_params`:

>>> search.best_pipeline_.describe_params() # doctest: +SKIP
{'α': 0.054...}

Validating hyperparameter search with nested cross-validation
-------------------------------------------------------------

To avoid overfitting hyperparameters, the best combination must be evaluated on
data that has not been used to select hyperparameters. This can be done with a
single train-test split or with nested cross-validation.

Single train-test split:

>>> split = pred.skb.train_test_split()
>>> search = pred.skb.get_randomized_search()
>>> search.fit(split['train'])
ParamSearch(expr=<Apply Ridge>,
            search=RandomizedSearchCV(estimator=None, param_distributions=None))
>>> search.score(split['test'])  # doctest: +SKIP
0.4922874902029253

For nested cross-validation we use :func:`skrub.cross_validate`, which accepts a
``pipeline`` parameter (as opposed to
:meth:`.skb.cross_validate() <Expr.skb.cross_validate>`
which always uses the default hyperparameters):

>>> skrub.cross_validate(pred.skb.get_randomized_search(), pred.skb.get_data())  # doctest: +SKIP
   fit_time  score_time  test_score
0  0.891390    0.002768    0.412935
1  0.889267    0.002773    0.519140
2  0.928562    0.003124    0.491722
3  0.890453    0.002732    0.428337
4  0.889162    0.002773    0.536168

Choices beyond estimator hyperparameters
----------------------------------------

Choices are not limited to scikit-learn hyperparameters: we can use choices
wherever we use expressions. The choice of the estimator to use, any argument of
an expression's method or :func:`deferred` function call, etc. can be replaced
with choices. We can also choose between several expressions to compare
different pipelines.

As an example of choices outside of scikit-learn estimators, we can consider
several ways to perform an aggregation on a pandas DataFrame:

>>> ratings = skrub.var("ratings")
>>> agg_ratings = ratings.groupby("movieId")["rating"].agg(
...     skrub.choose_from(["median", "mean"], name="rating_aggregation")
... )
>>> print(agg_ratings.skb.describe_param_grid())
- rating_aggregation: ['median', 'mean']

We can also choose between several completely different pipelines by turning a
choice into an expression, via its ``as_expr`` method (or by using
:func:`as_expr` on any object).

>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.ensemble import RandomForestRegressor

>>> data = skrub.var("data", diabetes_df)
>>> X = data.drop(columns="target", errors="ignore").skb.mark_as_X()
>>> y = data["target"].skb.mark_as_y()

>>> ridge_pred = X.skb.apply(skrub.optional(StandardScaler())).skb.apply(
...     Ridge(alpha=skrub.choose_float(0.01, 10.0, log=True, name="α")), y=y
... )
>>> rf_pred = X.skb.apply(
...     RandomForestRegressor(n_estimators=skrub.choose_int(5, 50, name="N 🌴")), y=y
... )
>>> pred = skrub.choose_from({"ridge": ridge_pred, "rf": rf_pred}).as_expr()
>>> print(pred.skb.describe_param_grid())
- choose_from({'ridge': …, 'rf': …}): 'ridge'
  optional(StandardScaler()): [StandardScaler(), None]
  α: choose_float(0.01, 10.0, log=True, name='α')
- choose_from({'ridge': …, 'rf': …}): 'rf'
  N 🌴: choose_int(5, 50, name='N 🌴')

Also note that as seen above, choices can be nested arbitrarily. For example it
is frequent to choose between several estimators, each of which contains choices
in its hyperparameters.

Linking choices
---------------

Choices can depend on another choice made with :func:`choose_from`,
:func:`choose_bool` or :func:`optional` through those objects' ``.match()``
method.

Suppose we want to use either ridge regression, random forest or gradient
boosting, and that we want to use imputation for ridge and random forest (only),
and scaling for the ridge (only). We can start by choosing the kind of
estimators and make further choices depend on the estimator kind:

>>> import skrub
>>> from sklearn.impute import SimpleImputer, KNNImputer
>>> from sklearn.preprocessing import StandardScaler, RobustScaler
>>> from sklearn.linear_model import Ridge
>>> from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

>>> estimator_kind = skrub.choose_from(
...     ["ridge", "random forest", "gradient boosting"], name="estimator"
... )
>>> imputer = estimator_kind.match(
...     {"gradient boosting": None},
...     default=skrub.choose_from([SimpleImputer(), KNNImputer()], name="imputer"),
... )
>>> scaler = estimator_kind.match(
...     {"ridge": skrub.choose_from([StandardScaler(), RobustScaler()], name="scaler")},
...     default=None,
... )
>>> predictor = estimator_kind.match(
...     {
...         "ridge": Ridge(),
...         "random forest": RandomForestRegressor(),
...         "gradient boosting": HistGradientBoostingRegressor(),
...     }
... )
>>> pred = skrub.X().skb.apply(imputer).skb.apply(scaler).skb.apply(predictor)
>>> print(pred.skb.describe_param_grid())
- estimator: 'ridge'
  imputer: [SimpleImputer(), KNNImputer()]
  scaler: [StandardScaler(), RobustScaler()]
- estimator: 'random forest'
  imputer: [SimpleImputer(), KNNImputer()]
- estimator: 'gradient boosting'

Note that only relevant choices are included in each subgrid. For example, when
the estimator is ``'random forest'``, the subgrid contains several options for
imputation but not for scaling.

In addition to ``match``, choices created with :func:`choose_bool` have an
``if_else()`` method which is a convenience helper equivalent to
``match({True: ..., False: ...})``.
