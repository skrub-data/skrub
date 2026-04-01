.. currentmodule:: skrub

.. _user_guide_data_ops_control_flow:

Running complex operations on DataOps variables: deferred evaluation
====================================================================

DataOps represent computations that have not been executed yet, and will
only be triggered when we call :meth:`.skb.eval() <DataOp.skb.eval>`, or when we
create the pipeline with :meth:`.skb.make_learner() <DataOp.skb.make_learner>` and
call one of its methods such as ``fit()``.

This means we cannot use standard Python control flow statements such as ``if``,
``for``, ``with``, etc. with DataOps, because those constructs would execute
immediately.

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
>>> for column in orders.columns:
...     pass
Traceback (most recent call last):
    ...
TypeError: This object is a DataOp that will be evaluated later, when your learner runs. So it is not possible to eagerly iterate over it now.

We get an error because the ``for`` statement tries to iterate immediately
over the columns. This is the way any computation on any variable is usually run,
referred to here as *eager* evaluation. However, ``orders.columns`` is not an actual
list of columns: it is a skrub DataOp that will produce a list of columns, later,
when we run the computation.

Therefore, we must delay the execution of the ``for`` statement until the computation
actually runs and ``orders.columns`` has been evaluated, hence the designation of
*deferred* computation rather than *eager*.

We can achieve this by defining a function that contains the control flow logic
we need, and decorating it with :func:`deferred`. This decorator defers the execution
of the function: when we call it, it does not run immediately. Instead, it returns
a skrub DataOp that wraps the function call. The original function is only
executed when the DataOp is evaluated, and will return the result as a DataOp.

>>> @skrub.deferred
... def with_upper_columns(df):
...     new_columns = [c.upper() for c in df.columns]
...     return df.set_axis(new_columns, axis="columns")

>>> with_upper_columns(orders)
<Call 'with_upper_columns'>
Result:
―――――――
   ITEM  PRICE  QTY
0   pen    1.5    1
1   cup    NaN    1
2   pen    1.5    2
3  fork    2.2    4

When the computation runs, ``orders`` will be evaluated first and the result (an
actual dataframe) will be passed as the ``df`` argument to our function. In practice,
the code inside a deferred function is completely equivalent to eager code, so
it is possible to use any Python control flow statement inside it, as well as
act on the data as if it were a regular DataFrame.

Within a function decorated with :func:`deferred`, objects are evaluated eagerly,
so it is possible to use standard Python control flow statements such as
``if``, ``for``, and it is possible to treat the inputs as if they were
regular objects (e.g., a Pandas DataFrame or Series).

A function can be marked as deferred using the :func:`deferred` decorator, but
also applied directly to a skrub DataOp using
:meth:`.skb.apply_func() <DataOp.skb.apply_func>`:

>>> def with_upper_columns(df):
...     new_columns = [c.upper() for c in df.columns]
...     return df.set_axis(new_columns, axis="columns")

>>> orders.skb.apply_func(with_upper_columns)
<Call 'with_upper_columns'>
Result:
―――――――
   ITEM  PRICE  QTY
0   pen    1.5    1
1   cup    NaN    1
2   pen    1.5    2
3  fork    2.2    4

It is also possible to create a specific deferred function from a preexisting eager
one: for instance, if we need to call module-level functions from a library. For
example, to delay the loading of a CSV file, we could write something like:

>>> csv_path = skrub.var("csv_path")
>>> data = skrub.deferred(pd.read_csv)(csv_path)

Another consequence of the fact that DataOps are evaluated lazily (we are
building a pipeline, not immediately computing a single result), any
transformation that we apply *must not modify its input in-place*, but leave it
unchanged and return a new value.

Finally, there are other situations where using :func:`deferred` can be helpful:

- When we have many nodes in our graph and want to collapse a sequence of steps into
  a single function call that appears as a single node.
- When certain function calls need to be deferred until the full computation
  runs, because they depend on the runtime environment, or on objects that
  cannot be pickled with the rest of the computation graph (for example, opening
  and reading a file).

.. rubric:: Examples

- See :ref:`sphx_glr_auto_examples_data_ops_1110_data_ops_intro.py` for an introductory
  example on how to use skrub DataOps on a single dataframe.
- See :ref:`sphx_glr_auto_examples_data_ops_1120_multiple_tables.py` for an example
  of how skrub DataOps can be used to process multiple tables using dataframe APIs.
- See :ref:`sphx_glr_auto_examples_data_ops_1130_choices.py` for an example of
  hyper-parameter tuning using skrub DataOps.
