.. currentmodule:: skrub

.. _skrub_pipeline_validation:

=====================================
Tuning and validating Skrub Pipelines
=====================================

Typically, we want to evaluate the prediction performance of our pipeline. This
is usually done by fitting it on a training dataset, then obtaining prediction
on an unseen, test dataset.

In scikit-learn, estimators and pipelines are given an ``X`` and a ``y`` matrix
with one row per observation from the very start. Therefore splitting of these
data into training and test set can be done independently from the pipeline.

However, in many real-world scenarios, our data sources are not already
organized into ``X`` and ``y`` matrices. Some transformations may be necessary to
build those, and we want to keep those transformations inside the pipeline so
that they can be reliably re-applied to new data.

Therefore, we must start our pipeline by creating the design matrix and targets,
then tell skrub which intermediate results in the pipeline constitute ``X`` and
``y`` respectively.

Let us consider a toy example where the only step needed to obtain ``X`` and
``y`` is extracting them from a single table -- in the input to the pipeline,
both are in the same dataframe. More complex transformations would be handled in
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

Rather than a single split, we often want to run cross-validation. The same
mechanism is used, except that several splits are done and the model fitting and
scoring is done for us. This is done with
:meth:`.skb.cross_validate() <Expr.skb.cross_validate>`.

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


We can then find the best hyperparameters.

>>> search = pred.skb.get_randomized_search(fitted=True, scoring="r2", random_state=0)
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


Choices are not limited to scikit-learn hyperparameters. The choice of the
estimator to use, any argument of an expression's method or deferred function
call, etc. can be replaced with choices. We can also choose between several
expressions to compare different pipelines. Choices can be nested arbitrarily.

>>> from sklearn.ensemble import RandomForestRegressor

>>> ridge = Ridge(alpha=skrub.choose_float(0.01, 10.0, log=True, name="α"))
>>> rf = RandomForestRegressor(n_estimators=skrub.choose_int(5, 50, name="N 🌴"))
>>> regressor = skrub.choose_from({"ridge": ridge, "rf": rf}, name="regressor")
>>> pred = X.skb.apply(regressor, y=y)
>>> print(pred.skb.describe_param_grid())
- regressor: 'ridge'
  α: choose_float(0.01, 10.0, log=True, name='α')
- regressor: 'rf'
  N 🌴: choose_int(5, 50, name='N 🌴')

>>> search = pred.skb.get_randomized_search(fitted=True, scoring="r2", random_state=0)
>>> search.results_ # doctest: +SKIP
   mean_test_score         α   N 🌴 regressor
0         0.480425  0.078092   NaN     ridge
1         0.477904  0.150784   NaN     ridge
2         0.470507  0.271016   NaN     ridge
3         0.443326  0.600529   NaN     ridge
4         0.439670  0.643117   NaN     ridge
5         0.423326       NaN  47.0        rf
6         0.416455       NaN  31.0        rf
7         0.403278       NaN  37.0        rf
8         0.348517       NaN   8.0        rf
9         0.168444  7.780156   NaN     ridge
