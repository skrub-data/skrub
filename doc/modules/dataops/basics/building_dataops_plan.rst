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
