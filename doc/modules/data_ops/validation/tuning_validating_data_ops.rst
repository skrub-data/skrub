.. _user_guide_data_ops_tuning_validating_dataops:

Tuning and validating Skrub DataOps plans
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
