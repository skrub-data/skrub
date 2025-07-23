.. _userguide_data_ops_ml_pipeline:
========================================================
Assembling Skrub DataOps into complex machine learning pipelines
========================================================

.. currentmodule:: skrub

Applying machine-learning estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to working directly with the API provided by the underlying data,
DataOps can also be used to apply machine-learning estimators from
scikit-learn or Skrub to the data. This is done through the
:meth:`.skb.apply() <DataOp.skb.apply>` method:

>>> import pandas as pd
>>> import skrub
>>> orders_df = pd.DataFrame(
...     {
...         "item": ["pen", "cup", "pen", "fork"],
...         "price": [1.5, None, 1.5, 2.2],
...         "qty": [1, 1, 2, 4],
...     }
... )
>>> orders = skrub.var("orders", orders_df)

>>> orders.skb.apply(skrub.TableVectorizer())
<Apply TableVectorizer>
Result:
―――――――
   item_cup  item_fork  item_pen  price  qty
0       0.0        0.0       1.0    1.5  1.0
1       1.0        0.0       0.0    NaN  1.0
2       0.0        0.0       1.0    1.5  2.0
3       0.0        1.0       0.0    2.2  4.0

It is also possible to apply a transformer to a subset of the columns:

>>> vectorized_orders = orders.skb.apply(
...     skrub.StringEncoder(n_components=3), cols="item"
... )
>>> vectorized_orders # doctest: +SKIP
<Apply StringEncoder>
Result:
―――――――
         item_0        item_1        item_2  price  qty
0  9.999999e-01  1.666000e-08  4.998001e-08    1.5    1
1 -1.332800e-07 -1.199520e-07  1.000000e+00    NaN    1
2  9.999999e-01  1.666000e-08  4.998001e-08    1.5    2
3  3.942477e-08  9.999999e-01  7.884953e-08    2.2    4

Then, we can export the transformation as a learner with
:meth:`.skb.make_learner() <DataOp.skb.make_learner>`

>>> pipeline = vectorized_orders.skb.make_learner(fitted=True)
>>> new_orders = pd.DataFrame({"item": ["fork"], "price": [2.2], "qty": [5]})
>>> pipeline.transform({"orders": new_orders}) # doctest: +SKIP
         item_0  item_1        item_2  price  qty
0  5.984116e-09     1.0 -1.323546e-07    2.2    5

Note that here the learner is fitted on the preview data, but in general it can
be exported without fitting, and then fitted on new data provided as an environment
dictionary.

Applying different transformers using Skrub selectors and DataOps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to use Skrub selectors to define which columns to apply
transformers to, and then apply different transformers to different subsets of
the data.

For example, this can be useful to apply :class:`~skrub.TextEncoder` to columns
that contain free-flowing text, and :class:`~skrub.StringEncoder` to other string
columns that contain categorical data such as country names.

In the example below, we apply a :class:`~skrub.StringEncoder` to columns
with high cardinality, a mathematical operation to columns with nulls, and a
:class:`~skrub.TableVectorizer` to all other columns.

>>> from skrub import selectors as s
>>> high_cardinality = s.string() - s.cardinality_below(2)
>>> has_nulls = s.has_nulls()
>>> leftover = s.all() - high_cardinality - has_nulls

>>> vectorizer = skrub.StringEncoder(n_components=2)
>>> vectorized_items = orders.skb.select(high_cardinality).skb.apply(vectorizer)
>>> vectorized_items # doctest: +SKIP
<Apply StringEncoder>
Result:
―――――――
          item_0        item_1  price  qty
0  1.511858e+00  9.380015e-08    1.5    1
1 -1.704687e-07  1.511858e+00    NaN    1
2  1.511858e+00  9.380015e-08    1.5    2
3 -5.458670e-09 -6.917769e-08    2.2    4

>>> vectorized_has_nulls = orders.skb.select(cols=has_nulls) * 11
>>> vectorized_has_nulls
    <BinOp: mul>
    Result:
    ―――――――
       price
    0   16.5
    1    NaN
    2   16.5
    3   24.2
>>> everything_else = orders.skb.select(cols=leftover).skb.apply(skrub.TableVectorizer())

After encoding the columns, the resulting DataOps can be concatenated together
to obtain the final result:

>>> encoded = (
...   everything_else.skb.concat([vectorized_items, vectorized_has_nulls], axis=1)
... )
>>> encoded # doctest: +SKIP
   qty        item_0        item_1  price
0  1.0  1.594282e+00 -1.224524e-07   16.5
1  1.0  9.228692e-08  1.473794e+00    NaN
2  2.0  1.594282e+00 -1.224524e-07   16.5
3  4.0  7.643604e-09  6.080018e-01   24.2

More info can be found in :ref:`userguide_selectors` and example
:ref:`sphx_glr_auto_examples_10_apply_on_cols.py`.


Documenting the DataOps plan with node names and descriptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can improve the readability of the DataOps plan by giving names and descriptions
to the nodes in the plan. This is done with :meth:`.skb.set_name() <DataOp.skb.set_name>`
and :meth:`.skb.set_description() <DataOp.skb.set_description>`.

>>> import skrub
>>> a = skrub.var('a', 1)
>>> b = skrub.var('b', 2)
>>> c = (a + b).skb.set_description('the addition of a and b')
>>> c.skb.description
'the addition of a and b'
>>> d = c.skb.set_name('d')
>>> d.skb.name
'd'

Both names and descriptions can be used to mark relevant parts of the learner, and
they can be accessed from the computational graph and the plan report.

Additionally, names can be used to bypass the computation of a node and override its
result by passing it as a key in the ``environment`` dictionary.

>>> e = d * 10
>>> e
<BinOp: mul>
Result:
―――――――
30
>>> e.skb.eval()
30
>>> e.skb.eval({'a': 10, 'b': 5})
150
>>> e.skb.eval({'d': -1}) # -1 * 10
-10

More info can be found in section :ref:`user_guide_truncating_dataplan_ref`.

Evaluating and debugging the DataOps plan with :meth:`.skb.full_report() <DataOp.skb.full_report>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All operations on DataOps are recorded in a computational graph, which can be
inspected with :meth:`.skb.full_report() <DataOp.skb.full_report>`. This method
generates a html report that shows the full plan, including all nodes,
their names, descriptions, and the transformations applied to the data.

An example of the report can be found
`here <../../_static/credit_fraud_report/index.html>`_.

For each node in the plan, the report shows:
- The name and the description of the node, if present.
- Predecessor and successor nodes in the computational graph.
- Where the code relative to the node is defined.
- The estimator fitted in the node along with its parameters (if applicable).
- The preview of the data at that node.

Additionally, if computations fail in the plan, the report shows the offending
node and the error message, which can help in debugging the plan.

By default, reports are saved in the ``skrub_data/execution_reports`` directory, but
they can be saved to a different location with the ``ouptut_dir`` parameter.
Note that the default path can be altered with the
``SKRUB_DATA_DIR`` environment variable. See :ref:`userguide_utils` for more details.

.. _user_guide_truncating_dataplan_ref:

Using only a part of a DataOps plan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Besides documenting a DataOps plan, the :meth:`.skb.set_name() <DataOp.skb.set_name>`
has additional functions. By setting a name, we can:

- Bypass the computation of that node and override its result by passing it as a
  key in the ``environment`` argument.
- Truncate the pipeline after this node to obtain the intermediate result with
  :meth:`SkrubLearner.truncated_after`.
- Retrieve that node and inspect the estimator that was fitted in it, if the
  node was created with :meth:`.skb.apply() <DataOp.skb.apply>`.

Here is a toy example with 3 steps:

>>> def load_data(url):
...     print("load: ", url)
...     return [1, 2, 3, 4]


>>> def transform(x):
...     print("transform")
...     return [item * 10 for item in x]


>>> def agg(x):
...     print("agg")
...     return max(x)


>>> url = skrub.var("url")
>>> output = (
...     url.skb.apply_func(load_data)
...     .skb.set_name("loaded")
...     .skb.apply_func(transform)
...     .skb.set_name("transformed")
...     .skb.apply_func(agg)
... )

Above, we give a name to each intermediate result with ``.skb.set_name()`` so
that we can later refer to it when manipulating a fitted pipeline.

>>> pipeline = output.skb.make_learner()
>>> pipeline.fit({"url": "file:///example.db"})
load:  file:///example.db
transform
agg
SkrubLearner(data_op=<Call 'agg'>)

>>> pipeline.transform({"url": "file:///example.db"})
load:  file:///example.db
transform
agg
40

Below, we bypass the data loading. Because we directly provide a value for the
intermediate result that we named ``"loaded"``, the corresponding computation is
skipped and the provided value is used instead. We can see that
``"load: ..."`` is not printed and that the rest of the computation proceeds
using ``[6, 5, 4]`` (instead of ``[1, 2, 3, 4]`` as before).

>>> pipeline.transform({"loaded": [6, 5, 4]})
transform
agg
60

Now we show how to stop at the result we named ``"transformed"``. With
``truncated_after``, we obtain a pipeline that computes that intermediate result
and returns it instead of applying the last transformation; note that ``"agg"``
is not printed and we get the output of ``transform()``, not of ``agg()``:

>>> truncated = pipeline.truncated_after("transformed")
>>> truncated.transform({"url": "file:///example.db"})
load:  file:///example.db
transform
[10, 20, 30, 40]


Subsampling data for easier development and debugging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the data used for the preview is large, it can be useful to work on a
subsample of the data to speed up the development and debugging process.
This can be done by calling the :meth:`.skb.subsample() <DataOp.skb.subsample>` method
on a variable: this signals to Skrub that what is shown when printing DataOps, or
returned by :meth:`.skb.preview() <DataOp.skb.preview>` is computed on a subsample
of the data.

Note that subsampling is "local": if it is applied to a variable, it only
affects the variable itself. This may lead to unexpected results and errors
if, for example, ``X`` is subsampled but ``y`` is not.

Subsampling **is turned off** by default when we call other methods such as
:meth:`.skb.eval() <DataOp.skb.eval>`,
:meth:`.skb.cross_validate() <DataOp.skb.cross_validate>`,
:meth:`.skb.train_test_split <DataOp.skb.train_test_split>`,
:meth:`DataOp.skb.make_learner`,
:meth:`DataOp.skb.make_randomized_search`, etc.
However, all of those methods have a ``keep_subsampling`` parameter that we can
set to ``True`` to force using the subsampling when we call them. Note that
even if we set ``keep_subsampling=True``, subsampling is not applied when using
``predict``.

See more details in a :ref:`full example <sphx_glr_auto_examples_data_ops_13_subsampling.py>`.
