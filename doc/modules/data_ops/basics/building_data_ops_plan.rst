.. currentmodule:: skrub

.. _user_guide_data_ops_plan:

Building a simple DataOps plan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's build a simple DataOps plan that adds two variables together.

We start by declaring the variables:

>>> import skrub

>>> a = skrub.var("a")
>>> b = skrub.var("b")

We then apply transformations (in this case, an addition) composing more complex DataOps.

>>> c = a + b
>>> c
<BinOp: add>

Finally, we can evaluate the plan by passing the **environment** in which the
plan should be evaluated. The environment is a dictionary that maps variable names
to their values.

>>> c.skb.eval({"a": 10, "b": 6})
16

As shown above, the special ``.skb`` attribute allows to interact with the DataOp
object itself, and :meth:`.skb.eval() <DataOp.skb.eval>` evaluates the DataOp plan.
By default, :meth:`.skb.eval() <DataOp.skb.eval>` uses the values passed in the
variable definitions, but it can also take an explicit environment
dictionary as an argument.


Finally, we can export the plan as a ``Learner`` that can be fitted and applied to
new data:

>>> learner = c.skb.make_learner()
>>> learner.fit_transform({"a": 10, "b": 7})
17

When using Data Ops, it is important to ensure that all operations are being tracked
by acting on the Data Ops, rather than (for example) the starting dataframe.
Consider the following example:

>>> import pandas as pd
>>> df = pd.DataFrame({"col": [1, 2, 3]})
>>> df
   col
0    1
1    2
2    3
>>> df_do = skrub.var("df", df)
>>> df_do
<Var 'df'>
Result:
―――――――
   col
0    1
1    2
2    3

``df_do`` is a Data Op that wraps ``df``, so its preview shows the content of ``df``.
Then, if we now modify ``df_do`` by doubling the column, we can see that both steps
(the creation of the variable, and the doubling) are now tracked by the final
Data Op.

>>> df_doubled = df_do.assign(col=df_do["col"]*2)
>>> df_doubled
<CallMethod 'assign'>
Result:
―――――――
   col
0    2
1    4
2    6
>>> print(df_doubled.skb.describe_steps())
Var 'df'
( Var 'df' )*
GetItem 'col'
BinOp: mul
CallMethod 'assign'
* Cached, not recomputed

On the other hand, working directly on ``df`` leads us to the same result, but
the actual operations are not being tracked.
By working only on Data Ops we ensure that all the operations done on the data
are added correctly to the computational graph, which then allows the resulting
learner to execute all steps as intended.

See :ref:`sphx_glr_auto_examples_data_ops_11_data_ops_intro.py` for an introductory
example on how to use skrub DataOps on a single dataframe.
