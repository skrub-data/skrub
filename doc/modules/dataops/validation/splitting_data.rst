.. _splitting_data:

Splitting the data in train and test sets
=========================================

We can use :meth:`.skb.train_test_split() <DataOp.skb.train_test_split>` to
perform a single train-test split. Skrub first evaluates the DataOps on
which we used :meth:`.skb.mark_as_X() <DataOp.skb.mark_as_X>` and
:meth:`.skb.mark_as_y() <DataOp.skb.mark_as_y>`: the first few steps of the
pipeline are executed until we have a value for ``X`` and for ``y``.
