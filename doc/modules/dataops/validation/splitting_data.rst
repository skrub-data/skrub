.. _splitting_data:

Splitting the data in train and test sets
=========================================

We can use :meth:`.skb.train_test_split() <DataOp.skb.train_test_split>` to
perform a single train-test split. Skrub first evaluates the DataOps on
which we used :meth:`.skb.mark_as_X() <DataOp.skb.mark_as_X>` and
:meth:`.skb.mark_as_y() <DataOp.skb.mark_as_y>`: the first few steps of the
pipeline are executed until we have a value for ``X`` and for ``y``.
Then, those
dataframes are split using the provided splitter function (by default
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

It is possible to define a custom splitter function to use instead of
:func:`sklearn.model_selection.train_test_split`.
