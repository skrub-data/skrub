.. _tuning_validating_dataops:

Tuning and validating Skrub DataOps plans
=========================================

To evaluate the prediction performance of our plan, we can fit it on a training
dataset, then obtaining prediction on an unseen, test dataset.

In scikit-learn, we pass to estimators and pipelines an ``X`` and ``y`` matrix
with one row per observation from the start. Therefore, we can split the
data into a training and test set independently from the pipeline.
