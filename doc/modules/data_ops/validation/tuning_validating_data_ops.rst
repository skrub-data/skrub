.. currentmodule:: skrub
.. _user_guide_data_ops_tuning_validating_dataops:

Tuning and validating skrub DataOps plans
=========================================

To evaluate the prediction performance of our plan, we can fit it on a training
dataset, then obtaining prediction on an unseen, test dataset.

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

We use :meth:`.skb.mark_as_X() <DataOp.skb.mark_as_X>` to indicate that this
intermediate result (the dataframe obtained after dropping "target") is the
``X`` design matrix. This is the dataframe that will be split into a training
and a testing part when we split our dataset or perform cross-validation.

Similarly for ``y``, we use :meth:`.skb.mark_as_y() <DataOp.skb.mark_as_y>`:

>>> y = data["target"].skb.mark_as_y()

Now we can add our supervised estimator:

>>> pred = X.skb.apply(Ridge(), y=y)
>>> pred # doctest: +SKIP
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

Improving the confidence in our score through cross-validation
==============================================================

We can increase our confidence in our score by using cross-validation instead of
a single split. The same mechanism is used but we now fit and evaluate the model
on several splits. This is done with :meth:`.skb.cross_validate()
<DataOp.skb.cross_validate>`.

>>> pred.skb.cross_validate() # doctest: +SKIP
   fit_time  score_time  test_score
0  0.002816    0.001344    0.321665
1  0.002685    0.001323    0.440485
2  0.002468    0.001308    0.422104
3  0.002748    0.001321    0.424661
4  0.002649    0.001309    0.441961

.. _user_guide_data_ops_splitting_data:

Splitting the data in train and test sets
=========================================

We can use :meth:`.skb.train_test_split() <DataOp.skb.train_test_split>` to
perform a single train-test split. skrub first evaluates the DataOps on
which we used :meth:`.skb.mark_as_X() <DataOp.skb.mark_as_X>` and
:meth:`.skb.mark_as_y() <DataOp.skb.mark_as_y>`: the first few steps of the
pipeline are executed until we have a value for ``X`` and for ``y``.
Then, those
dataframes are split using the provided split function (by default
scikit-learn's :func:`sklearn.model_selection.train_test_split`).

>>> split = pred.skb.train_test_split(shuffle=False)
>>> split.keys()
dict_keys(['train', 'test', 'X_train', 'X_test', 'y_train', 'y_test'])

``train`` and ``test`` are the full dictionaries corresponding to the training
and testing data. The corresponding ``X`` and ``y`` are the values, in those
dictionaries, for the nodes marked with
:meth:`.skb.mark_as_X() <DataOp.skb.mark_as_X>`
and :meth:`.skb.mark_as_y() <DataOp.skb.mark_as_y>`.

We can now fit our pipeline on the training data:

>>> learner = pred.skb.make_learner()
>>> learner.fit(split["train"])
SkrubLearner(data_op=<Apply Ridge>)

Only the training part of ``X`` and ``y`` are used. The subsequent steps are
evaluated, using this data, to fit the rest of the pipeline.

And we can obtain predictions on the test part:

>>> test_pred = learner.predict(split["test"])
>>> test_y_true = split["y_test"]

>>> from sklearn.metrics import r2_score

>>> r2_score(test_y_true, test_pred) # doctest: +SKIP
0.440999149220359

It is possible to define a custom split function to use instead of
:func:`sklearn.model_selection.train_test_split`.

Passing additional arguments to the splitter
============================================

Sometimes we want to pass additional data to the cross-validation splitter.

For example, if there is a group structure in our data (such as sites,
hospitals, etc.) and we want the model to generalize to unseen groups, we must
ensure while evaluating it that each group goes entirely in the train set or the
test set, but is not divided among the 2. This can be done with
:class:`sklearn.model_selection.GroupKFold`,
:class:`sklearn.model_selection.LeavePGroupsOut`, etc. . The ``split`` function
of those objects accepts a ``groups`` parameter. We can compute the groups
inside of the DataOp and pass them to :meth:`DataOp.skb.mark_as_X` and they will
be passed to the splitter.

>>> df = skrub.datasets.toy_products()
>>> df
   description  price            seller     category
0       screen    100   supermarket.com  electronics
1       hammer     15  bestproducts.com        tools
2     keyboard     20   supermarket.com  electronics
3      usb key      9  bestproducts.com  electronics
4      charger     13  bestproducts.com  electronics
5  screwdriver     12   supermarket.com        tools

Suppose we want to assess generalization to new sellers. While splitting for
cross-validation we must group products by seller. We do it with
:class:`sklearn.model_selection.LeaveOneGroupOut`.

>>> from sklearn.dummy import DummyClassifier
>>> from sklearn.model_selection import LeaveOneGroupOut

>>> data = skrub.var("df", df)
>>> groups = data["seller"]
>>> X = data[["description", "price"]].skb.mark_as_X(
...     cv=LeaveOneGroupOut(), split_kwargs={"groups": groups}
... )
>>> y = data["category"].skb.mark_as_y()
>>> pred = X.skb.apply(DummyClassifier(), y=y)
>>> split = pred.skb.train_test_split()

The train set only contains data from the "supermarket.com" seller.

>>> split["X_train"]
   description  price
0       screen    100
2     keyboard     20
5  screwdriver     12

The test set only contains data from the "bestproducts.com" seller.

>>> split["X_test"]
  description  price
1      hammer     15
3     usb key      9
4     charger     13

Passing additional arguments to the scorer
==========================================

Sometimes we have additional information to pass to the scorer such as sample
weights, group information etc.

We can control how scoring is performed by using
:meth:`DataOp.skb.with_scoring`. It has a ``scoring`` parameter, which can be
anything scikit-learn's :func:`~sklearn.model_selection.cross_validate` accepts
for ``scoring`` such as a metric name, callable scorer, or dict mapping metric
names to scorers (see the reference documentation of
:meth:`DataOp.skb.with_scoring` for details).

It also accepts a ``kwargs`` argument, which are passed to the scorer when
evaluating the learner.

Importantly, the ``scoring`` and ``kwargs`` can be DataOps, which will be
computed when scoring the learner -- so for example, sample weights can be
computed dynamically.

Using the same toy dataset as above, suppose we want to give more weight to more
expensive products:

>>> X = data[["description", "price"]].skb.mark_as_X(cv=2)
>>> y = data["category"].skb.mark_as_y()
>>> pred = X.skb.apply(DummyClassifier(), y=y)

The default score is the (unweighted) accuracy:

>>> pred.skb.cross_validate() # doctest: +SKIP
   fit_time  score_time  test_score
0  0.003982    0.002405    0.666667
1  0.002582    0.002169    0.666667

We set the scoring to provide the sample weights:

>>> sample_weight = X["price"]
>>> pred.skb.with_scoring(
...     "accuracy", kwargs={"sample_weight": sample_weight}
... ).skb.cross_validate() # doctest: +SKIP
   fit_time  score_time  test_accuracy
0  0.003045    0.003275       0.888889
1  0.002659    0.003026       0.647059

Besides passing extra arguments, :meth:`DataOp.skb.with_scoring` can also be
useful to control what should be used as the default scoring metric for our
learner, just as the ``cv`` parameter of :meth:`DataOp.skb.mark_as_X` defines
the default cross-validation splitting strategy.

>>> split = pred.skb.train_test_split()
>>> learner = pred.skb.with_scoring('neg_log_loss').skb.make_learner()
>>> learner.fit(split['train'])
SkrubLearner(data_op=<Scoring <Apply DummyClassifier> (1 scorers)>
    This DataOp will be scored with:
      - 'neg_log_loss'
    Use .skb.cross_validate(…) or .skb.make_learner(…).score(…) to compute scores.)
>>> learner.score(split['test']) # doctest: +SKIP
-0.6365141682948128

Note that the score above is negative: it is the negative log loss we passed to
``with_scoring``, and not the default score (accuracy, which would be positive).

:meth:`DataOp.skb.with_scoring` only changes how scoring is performed
(the outputs of :meth:`DataOp.skb.cross_validate`,
:meth:`DataOp.skb.make_randomized_search`, :class:`SkrubLearner.score <SkrubLearner>` etc.),
**not** the actual outputs of the learner (it does _not_ affect the outputs of
:meth:`DataOp.skb.eval`, :class:`SkrubLearner.predict <SkrubLearner>`, etc.)

This method can be called several times to add scorers that take different
kwargs. See the reference documentation for details.
