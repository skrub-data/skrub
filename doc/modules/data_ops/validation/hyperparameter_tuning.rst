.. currentmodule:: skrub
.. _user_guide_data_ops_hyperparameter_tuning:

Hyperparameter tuning on a DataOps learner
==========================================


The skrub ``choose_*`` tools
----------------------------

skrub provides a convenient way to declare ranges of possible values, and tune
those choices to keep the values that give the best predictions on a validation
set.

Rather than specifying a grid of hyperparameters separately from the pipeline,
we simply insert special skrub objects in place of the value.

We define the same set of operations as before:

>>> from sklearn.datasets import load_diabetes
>>> from sklearn.linear_model import Ridge
>>> import skrub
>>> diabetes_df = load_diabetes(as_frame=True)["frame"]
>>> data = skrub.var("data", diabetes_df)
>>> X = data.drop(columns="target", errors="ignore").skb.mark_as_X()
>>> y = data["target"].skb.mark_as_y()
>>> pred = X.skb.apply(Ridge(), y=y)

Now, we can
replace the hyperparameter ``alpha`` (which should be a float) with a range
created by :func:`skrub.choose_float`. skrub can use it to select the best value
for ``alpha``.



>>> pred = X.skb.apply(
...     Ridge(alpha=skrub.choose_float(1e-6, 10.0, log=True, name="α")), y=y
... )

.. warning::

   When we do :meth:`.skb.make_learner() <DataOp.skb.make_learner>`, the
   pipeline we obtain does not perform any hyperparameter tuning. The pipeline
   we obtain by default uses default values for each of the choices. For numeric
   choices it is the middle of the range (unless an explicit default has been
   set when creating the choice), and for :func:`choose_from` it is the first
   option we give it. We can also obtain random choices, or choices suggested by
   an Optuna :class:`trial <optuna.trial.Trial>`, by passing the ``choose``
   parameter.

   To get a pipeline that runs an internal cross-validation to select the best
   hyperparameters, we must use :meth:`.skb.make_grid_search()
   <DataOp.skb.make_grid_search()>` or :meth:`.skb.make_randomized_search()
   <DataOp.skb.make_randomized_search>`. We can also use `Optuna
   <https://optuna.readthedocs.io/>`_ to choose the best hyperparameters as shown
   in :ref:`this example <example_optuna_choices>`.


Skrub provides over 10 different ``choose`` methods for tuning use cases, all detailed
in :ref:`here <choice_list>`

.. _choice-defaults-table:

.. list-table:: Default choice outcomes
   :header-rows: 1

   * - Choosing function
     - Description
     - Default outcome
   * - :func:`choose_from([10, 20]) <choose_from>`
     - Choose between the listed options (10 and 20).
     - First outcome in the list: ``10``
   * - :func:`choose_from({"a_name": 10, "b_name": 20}) <choose_from>`
     - Choose between the listed options (10 and 20). Dictionary keys serve as
       names for the options.
     - First outcome in the dictionary: ``10``
   * - :func:`optional(10) <optional>`
     - Choose between the provided value and ``None`` (useful for optional
       transformations in a pipeline, e.g., ``optional(StandardScaler())``).
     - The provided ``value``: ``10``
   * - :func:`choose_bool() <choose_bool>`
     - Choose between True and False.
     - ``True``
   * - :func:`choose_float(1.0, 100.0) <choose_float>`
     - Sample a floating-point number in a range.
     - The middle of the range: ``50.5``
   * - :func:`choose_int(1, 100) <choose_int>`
     - Sample an integer in a range.
     - The integer closest to the middle of the range: ``50``
   * - :func:`choose_float(1.0, 100.0, log=True) <choose_float>`
     - Sample a float in a range on a logarithmic scale.
     - The middle of the range on a log scale: ``10.0``
   * - :func:`choose_int(1, 100, log=True) <choose_int>`
     - Sample an integer in a range on a logarithmic scale.
     - The integer closest to the middle of the range on a log scale: ``10``
   * - :func:`choose_float(1.0, 100.0, n_steps=4) <choose_float>`
     - Sample a float on a grid.
     - The step closest to the middle of the range: ``34.0`` (steps: ``[1.0, 34.0, 67.0, 100.0]``)
   * - :func:`choose_int(1, 100, n_steps=4) <choose_int>`
     - Sample an integer on a grid.
     - The step closest to the middle of the range: ``34`` (steps: ``[1, 34, 67, 100]``)
   * - :func:`choose_float(1.0, 100.0, log=True, n_steps=4) <choose_float>`
     - Sample a float on a logarithmically spaced grid.
     - The step closest to the middle of the range on a log scale: ``4.64``
       (steps: ``[1.0, 4.64, 21.54, 100.0]``)
   * - :func:`choose_int(1, 100, log=True, n_steps=4) <choose_int>`
     - Sample an integer on a logarithmically spaced grid.
     - The step closest to the middle of the range on a log scale: ``5``
       (steps: ``[1, 5, 22, 100]``)


The default choices for a DataOp, those that get used when calling
:meth:`.skb.make_learner() <DataOp.skb.make_learner>`, can be inspected with
:meth:`.skb.describe_defaults() <DataOp.skb.describe_defaults>`:

>>> pred.skb.describe_defaults()
{'α': 0.00316...}

We can then find the best hyperparameters.

>>> search = pred.skb.make_randomized_search(fitted=True)
>>> search.results_  # doctest: +SKIP
          α  mean_test_score
0  0.000480         0.482327
1  0.000287         0.482327
2  0.000014         0.482317
3  0.000012         0.482317
4  0.000006         0.482317
5  0.134157         0.478651
6  0.249613         0.472019
7  0.612327         0.442312
8  2.664713         0.308492
9  3.457901         0.275007

A human-readable description of parameters for a pipeline can be obtained with
:meth:`SkrubLearner.describe_params`:

>>> search.best_learner_.describe_params()  # doctest: +SKIP
{'α': 0.000479...}

It is also possible to use :meth:`ParamSearch.plot_results` to visualize the results
of the search using a parallel coordinates plot.

This could also be done with Optuna, either by passing ``backend='optuna'`` to
:meth:`DataOp.skb.make_randomized_search`, or by using Optuna directly:

>>> import optuna  # doctest: +SKIP
>>> def objective(trial):   # doctest: +SKIP
...     learner = pred.skb.make_learner(choose=trial)
...     cv_results = skrub.cross_validate(learner, pred.skb.get_data())
...     return cv_results['test_score'].mean()
>>> study = optuna.create_study(direction="maximize")   # doctest: +SKIP
>>> study.optimize(objective, n_trials=10)   # doctest: +SKIP
>>> best_learner = pred.skb.make_learner(choose=study.best_trial)   # doctest: +SKIP
>>> best_learner.describe_params()  # doctest: +SKIP
{'α': 0.0006391165935023005}


Rather than fitting a randomized or grid search to find the best combination, it
is also possible to obtain an iterator over different parameter combinations to
inspect their outputs or to have manual control over the model selection. This can
be done with :meth:`.skb.iter_learners_grid() <DataOp.skb.iter_learners_grid>` or
:meth:`.skb.iter_learners_randomized() <DataOp.skb.iter_learners_randomized>` (
which yield the candidate pipelines that are explored by the grid and randomized
search respectively), or with the ``choose`` parameter of
:meth:`.skb.make_learner() <DataOp.skb.make_learner>`.

A full example of how to use hyperparameter search is available in
:ref:`sphx_glr_auto_examples_data_ops_1130_choices.py`, and a full example using
Optuna is in :ref:`example_optuna_choices`.

|


.. _user_guide_data_ops_feature_selection:

Feature selection with skrub :class:`SelectCols` and :class:`DropCols`
----------------------------------------------------------------------

It is possible to combine :class:`SelectCols` and :class:`DropCols` with
:func:`choose_from` to perform feature selection by dropping specific columns
and evaluating how this affects the downstream performance.

Consider this example. We first define the variable:

>>> import pandas as pd
>>> import skrub.selectors as s
>>> from sklearn.preprocessing import StandardScaler, OneHotEncoder
>>> df = pd.DataFrame({"text": ["foo", "bar", "baz"], "number": [1, 2, 3]})
>>> X = skrub.X(df)

Then, we use the :ref:`skrub selectors <user_guide_selectors>` to encode each
column with a different transformer:

>>> X_enc = X.skb.apply(StandardScaler(), cols=s.numeric()).skb.apply(
...     OneHotEncoder(sparse_output=False), cols=s.string()
... )
>>> X_enc
<Apply OneHotEncoder>
Result:
―――――――
     number  text_bar  text_baz  text_foo
0 -1.224745       0.0       0.0       1.0
1  0.000000       1.0       0.0       0.0
2  1.224745       0.0       1.0       0.0

Now we can use :class:`skrub.DropCols` to define two possible selection strategies:
first, we drop the column ``number``, then we drop all columns that start with
``text``. We rely again on the skrub selectors for this:

>>> from skrub import DropCols
>>> drop = DropCols(cols=skrub.choose_from(
...     {"number": s.cols("number"), "text": s.glob("text_*")})
... )
>>> X_enc.skb.apply(drop)
<Apply DropCols>
Result:
―――――――
   text_bar  text_baz  text_foo
0       0.0       0.0       1.0
1       1.0       0.0       0.0
2       0.0       1.0       0.0

We can see the generated parameter grid with :func:`DataOps.skb.describe_param_grid()`.

>>> X_enc.skb.apply(drop).skb.describe_param_grid()
"- choose_from({'number': …, 'text': …}): ['number', 'text']\n"

A more advanced application of this technique is used in
`this tutorial on forecasting timeseries <https://skrub-data.org/EuroSciPy2025/content/notebooks/single_horizon_prediction.html>`_,
along with the feature engineering required to prepare the columns, and the
analysis of the results.


Validating hyperparameter search with nested cross-validation
-------------------------------------------------------------

To avoid overfitting hyperparameters, the best combination must be evaluated on
data that has not been used to select hyperparameters. This can be done with a
single train-test split or with nested cross-validation.

Using the same examples as the previous sections:

>>> from sklearn.datasets import load_diabetes
>>> from sklearn.linear_model import Ridge
>>> import skrub
>>> diabetes_df = load_diabetes(as_frame=True)["frame"]
>>> data = skrub.var("data", diabetes_df)
>>> X = data.drop(columns="target", errors="ignore").skb.mark_as_X()
>>> y = data["target"].skb.mark_as_y()
>>> pred = X.skb.apply(
...     Ridge(alpha=skrub.choose_float(0.01, 10.0, log=True, name="α")), y=y
... )

Single train-test split:

>>> split = pred.skb.train_test_split()
>>> search = pred.skb.make_randomized_search()
>>> search.fit(split['train'])
ParamSearch(data_op=<Apply Ridge>,
            search=RandomizedSearchCV(estimator=None, param_distributions=None))
>>> search.score(split['test'])  # doctest: +SKIP
0.4922874902029253

For nested cross-validation we use :func:`skrub.cross_validate`, which accepts a
``pipeline`` parameter (as opposed to
:meth:`.skb.cross_validate() <DataOp.skb.cross_validate>`
which always uses the default hyperparameters):

>>> skrub.cross_validate(pred.skb.make_randomized_search(), pred.skb.get_data())  # doctest: +SKIP
   fit_time  score_time  test_score
0  0.891390    0.002768    0.412935
1  0.889267    0.002773    0.519140
2  0.928562    0.003124    0.491722
3  0.890453    0.002732    0.428337
4  0.889162    0.002773    0.536168

Going beyond estimator hyperparameters: nesting choices and choosing pipelines
------------------------------------------------------------------------------

Choices are not limited to scikit-learn hyperparameters: we can use choices
wherever we use DataOps. The choice of the estimator to use, any argument of
a DataOp's method or :func:`deferred` function call, etc. can be replaced
with choices. We can also choose between several DataOps to compare
different pipelines.

As an example of choices outside of scikit-learn estimators, we can consider
several ways to perform an aggregation on a pandas DataFrame:

>>> import skrub
>>> ratings = skrub.var("ratings")
>>> agg_ratings = ratings.groupby("movieId")["rating"].agg(
...     skrub.choose_from(["median", "mean"], name="rating_aggregation")
... )
>>> print(agg_ratings.skb.describe_param_grid())
- rating_aggregation: ['median', 'mean']

We can also choose between several completely different pipelines by turning a
choice into a DataOp, via its ``as_data_op`` method (or by using
:func:`as_data_op` on any object).

>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.datasets import load_diabetes
>>> from sklearn.linear_model import Ridge
>>> import skrub
>>> diabetes_df = load_diabetes(as_frame=True)["frame"]
>>> data = skrub.var("data", diabetes_df)
>>> X = data.drop(columns="target", errors="ignore").skb.mark_as_X()
>>> y = data["target"].skb.mark_as_y()

>>> ridge_pred = X.skb.apply(skrub.optional(StandardScaler())).skb.apply(
...     Ridge(alpha=skrub.choose_float(0.01, 10.0, log=True, name="α")), y=y
... )
>>> rf_pred = X.skb.apply(
...     RandomForestRegressor(n_estimators=skrub.choose_int(5, 50, name="N 🌴")), y=y
... )
>>> pred = skrub.choose_from({"ridge": ridge_pred, "rf": rf_pred}).as_data_op()
>>> print(pred.skb.describe_param_grid())
- choose_from({'ridge': …, 'rf': …}): 'ridge'
  optional(StandardScaler()): [StandardScaler(), None]
  α: choose_float(0.01, 10.0, log=True, name='α')
- choose_from({'ridge': …, 'rf': …}): 'rf'
  N 🌴: choose_int(5, 50, name='N 🌴')

Also note that as seen above, choices can be nested arbitrarily. For example it
is frequent to choose between several estimators, each of which contains choices
in its hyperparameters.


Linking choices depending on other choices
------------------------------------------

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
