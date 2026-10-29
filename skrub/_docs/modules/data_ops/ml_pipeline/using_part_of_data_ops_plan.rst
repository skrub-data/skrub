.. currentmodule:: skrub
.. _user_guide_data_ops_truncating_dataplan:

Using only a part of a DataOps plan
===================================

Besides documenting a DataOps plan, the :meth:`.skb.set_name() <DataOp.skb.set_name>`
has additional functions. By setting a name, we can:

- Bypass the computation of that node and override its result by passing it as a
  key in the ``environment`` argument.
- Truncate the computational graph after this node to obtain the intermediate result with
  :meth:`SkrubLearner.truncated_after`.
- Retrieve that node and inspect the estimator that was fitted in it, if the
  node was created with :meth:`.skb.apply() <DataOp.skb.apply>`.

Here is a toy example with 4 steps:

>>> def load_data(url):
...     print("load: ", url)
...     return [1, 2, 3, 4]


>>> def transform(x):
...     print("transform")
...     return [item * 10 for item in x]


>>> def agg(x):
...     print("agg")
...     return max(x)


>>> import skrub
>>> url = skrub.var("url")
>>> output = (
...     url.skb.apply_func(load_data)
...     .skb.set_name("loaded")
...     .skb.apply_func(transform)
...     .skb.set_name("transformed")
...     .skb.apply_func(agg)
... )

Above, we give a name to each intermediate result with ``.skb.set_name()`` so
that we can later refer to it when manipulating a fitted learner.

>>> learner = output.skb.make_learner()
>>> learner.fit({"url": "file:///example.db"})
load:  file:///example.db
transform
agg
SkrubLearner(data_op=<Call 'agg'>)

>>> learner.transform({"url": "file:///example.db"})
load:  file:///example.db
transform
agg
40

Below, we bypass the data loading. Because we directly provide a value for the
intermediate result that we named ``"loaded"``, the corresponding computation is
skipped and the provided value is used instead. We can see that
``"load: ..."`` is not printed and that the rest of the computation proceeds
using ``[6, 5, 4]`` (instead of ``[1, 2, 3, 4]`` as before).

>>> learner.transform({"loaded": [6, 5, 4]})
transform
agg
60

Now we show how to stop at the result we named ``"transformed"``. With
``truncated_after``, we obtain a learner that computes that intermediate result
and returns it instead of applying the last transformation; note that ``"agg"``
is not printed and we get the output of ``transform()``, not of ``agg()``:

>>> truncated = learner.truncated_after("transformed")
>>> truncated.transform({"url": "file:///example.db"})
load:  file:///example.db
transform
[10, 20, 30, 40]
