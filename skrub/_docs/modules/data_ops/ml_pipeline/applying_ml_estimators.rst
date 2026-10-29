.. currentmodule:: skrub
.. _user_guide_data_ops_applying_ml_estimators:

Applying machine-learning estimators
=====================================

In addition to working directly with the API provided by the underlying data,
DataOps can also be used to apply machine-learning estimators from
scikit-learn or skrub to the data. This is done through the
:func:`.skb.apply() <DataOp.skb.apply>` method:

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

It is also possible to apply a transformer to a subset of the columns. The ``cols``
parameter can also use a skrub :ref:`selector <user_guide_selectors>` for finer
grained control.
Note that any column that is not selected is passed through unchanged, like below:

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

>>> learner = vectorized_orders.skb.make_learner(fitted=True)
>>> new_orders = pd.DataFrame({"item": ["fork"], "price": [2.2], "qty": [5]})
>>> learner.transform({"orders": new_orders}) # doctest: +SKIP
         item_0  item_1        item_2  price  qty
0  5.984116e-09     1.0 -1.323546e-07    2.2    5

Note that here the learner is **fitted** on the preview data, but in general it can
be exported without fitting, and then fitted on new data provided as an environment
dictionary. By default, the learner is returned without fitting.

>>> learner = vectorized_orders.skb.make_learner()
>>> learner.fit({"orders": orders_df})
SkrubLearner(data_op=<Apply StringEncoder>)
