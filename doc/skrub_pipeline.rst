.. _skrub_pipeline:

===================================================
Skrub Pipeline: flexible machine learning pipelines
===================================================

.. currentmodule:: skrub

Introduction
~~~~~~~~~~~~

Skrub provides an easy way to build complex, flexible machine learning pipelines.
There are several problems that are not easily addressed with standard scikit-learn
tools such as :class:`~sklearn.pipeline.Pipeline` and
:class:`~sklearn.compose.ColumnTransformer`, and for which a Skrub pipeline offers a
solution:

- **Multiple tables**: When there are several tables of different shapes (for example,
  "Customers", "Orders", and "Products" tables), standard scikit-learn estimators fall
  short, as they expect a single design matrix ``X`` and a target array ``y``, with one
  row per observation.
- **DataFrame wrangling**: Performing typical DataFrame operations such as projections,
  joins, and aggregations—leveraging the powerful APIs of pandas or polars—is not
  easily expressed in standard pipelines.
- **Iterative development**: Building a pipeline step by step while inspecting
  intermediate results allows for a short feedback loop and early discovery of errors.
- **Hyperparameter tuning**: Choices of estimators, hyperparameters, and even the
  pipeline architecture can be guided by validation scores. Specifying a grid of
  hyperparameters separately from the model (as in
  :class:`~sklearn.model_selection.GridSearchCV`) becomes difficult in complex
  pipelines.

What is the difference with scikit-learn :class:`~sklearn.pipeline.Pipeline`?
=============================================================================

Scikit-learn pipelines represent a linear sequence of transformations on one
table with a fixed number of rows.

.. image:: ../../_static/sklearn_pipeline.svg
    :width: 500

Skrub expressions, on the other hand, can manipulate any number of variables.
The transformation they perform is not a linear sequence but any Directed
Acyclic Graph of computations.

.. image:: ../../_static/skrub_expressions.svg

What is the difference with orchestrators like Apache Airflow?
==============================================================

Skrub pipelines are not an orchestrator, and don't offer capabilities for scheduling
runs or provisionning resources and environments. They are a generalization of
scikit-learn pipelines, which can still be used within an orchestrator.

.. _expressions:

Skrub expressions
~~~~~~~~~~~~~~~~~

Skrub pipelines are built using special objects that represent intermediate results in a
computation. They record the different operations we perform on them (such as
applying operators or calling methods), which allows to later retrieve the
whole computation graph as a machine-learning estimator that can be fitted and
applied to unseen data.

Because those Skrub objects encapsulate a computation, which can be evaluated
to produce a result, we call them "expressions".

The simplest expressions are variables, which represent inputs to our
machine-learning estimator such as a ``"products"`` or a ``"customers"`` tables
or dataframes.

Those variables can be combined with operators and function calls to build up
more complex expressions. The estimator is being constructed implicitly when we
apply those operations, rather than by providing an explicit list of
transformations.

We start by declaring inputs:

.. code:: python

    import skrub

    a = skrub.var("a")
    b = skrub.var("b")

We then apply transformations, which we can finally evaluate, by passing a dictionary
mapping input name to values:

.. code:: python

    c = a + b
    c.skb.eval({"a": 10, "b": 6})
    # 16

As shown above, the special ``.skb`` attribute allows to interact with the expression
object itself, and ``.skb.eval()`` evaluate an expression.

Access to any other attribute is simply added as a new operation in the computation
graph:

.. code:: python

    d = c.capitalize()
    d.skb.eval({"a": "hello, ", "b": "world!"})
    # Hello world!

Finally, we can get an estimator that can be fitted and applied to data.

.. code:: python

    estimator = c.skb.get_estimator()
    estimator.fit_transform({"a": 10, "b": 7})
    # 17

Previews
~~~~~~~~

As we saw above, we can call ``.skb.eval()`` with a dictionary of bindings to compute
the result of the pipeline. However, to develop interactively without having to call
``.skb.eval()`` repeatedly, skrub provides a way to preview the result of the
expression. When we create a variable, if we pass a value in addition to its name,
skrub will use it to compute the result of the expression on that value.

.. code:: python

   a = skrub.var("a", 10)
   b = skrub.var("b", 6)
   c = a + b
   c  # we don't need to call .skb.eval anymore!
   # 16

Note that example values are immutable throughout the pipeline. This means that
to change the value of a variable, we need to create the pipeline again with
the new value.

Composing expressions
~~~~~~~~~~~~~~~~~~~~~

Suppose we want to process dataframes that look like this:

.. code:: python

    import pandas as pd

    orders_df = pd.DataFrame(
        {
            "item": ["pen", "cup", "pen", "fork"],
            "price": [1.5, None, 1.5, 2.2],
            "qty": [1, 1, 2, 4],
        }
    )

We can create a skrub variable to represent that input:

.. code:: python

    orders = skrub.var("orders", orders_df)


Because we know we will feed a DataFrame to the computation, we manipulate
``orders`` as if it were a DataFrame.

We can access its attributes:

.. code:: python

    orders.columns
    # Index([item, price, qty])

    orders["item"].iloc[1:]
    # 1    cup
    # 2    pen
    # 3    fork
    # Name: item, dtype: object

We can apply operators:

.. code:: python

    orders["price"] * orders["qty"]
    # 0    1.5
    # 1    NaN
    # 2    3.0
    # 3    8.8
    # dtype: float64

Calling methods:

.. code:: python

    orders.assign(total=orders["price"] * orders["qty"])
    #    item  price  qty  total
    # 0   pen    1.5    1    1.5
    # 1   cup    NaN    1    NaN
    # 2   pen    1.5    2    3.0
    # 3  fork    2.2    4    8.8

It is important to note that the original ``orders`` pipeline is not modified
by the previous cells. Instead, in each cell a new expression is created that
represents the result of the operation.

This is similar to how Pandas and Polars
dataframes work: when we call a method on a dataframe, it returns a new
dataframe that represents the result of the operation, rather than modifying
the original dataframe in place. However, while in Pandas it is possible to
work on a dataframe in place, Skrub does not allow this.

Applying machine-learning estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned above, in addition to those usual operations, the expressions
have a special attribute: ``.skb``, which gives access to the methods and objects
provided by Skrub. A particularly important one is
``.skb.apply()``, which allows to scikit-learn estimators to the pipeline.

.. code:: python

    orders.skb.apply(skrub.TableVectorizer())
    # <Apply TableVectorizer>
    # Result:
    # ―――――――
    #    item_cup  item_fork  item_pen  price  qty
    # 0       0.0        0.0       1.0    1.5  1.0
    # 1       1.0        0.0       0.0    NaN  1.0
    # 2       0.0        0.0       1.0    1.5  2.0
    # 3       0.0        1.0       0.0    2.2  4.0

It is also possible to apply a transformer to a subset of the columns:

.. code:: python

    vectorized_orders = orders.skb.apply(
        skrub.StringEncoder(n_components=3), cols="item"
    )
    vectorized_orders
    # <Apply StringEncoder>
    # Result:
    # ―――――――
    #          item_0        item_1        item_2  price  qty
    # 0  1.000000e+00  4.260130e-08 -2.691092e-09    1.5    1
    # 1 -1.703285e-07  9.999984e-01  1.792236e-03    NaN    1
    # 2  1.000000e+00  4.260130e-08 -2.691092e-09    1.5    2
    # 3  9.629613e-10 -1.792075e-03  9.999982e-01    2.2    4

Again, the crucial part is that when applying such operations, the return
value encapsulates the whole computation that produces the result we see. We are
not interested in a single result for the example values we provided, but in
the ability to retrieve a machine-learning estimator that we can fit and then
apply to unseen data.

We can retrieve the estimator, fit it on the data we provided initially, then apply
it on new data:

.. code:: python

    estimator = vectorized_orders.skb.get_estimator(fitted=True)
    new_orders = pd.DataFrame({"item": ["fork"], "price": [2.2], "qty": [5]})
    estimator.transform({"orders": new_orders})
    #          item_0    item_1    item_2  price  qty
    # 0  3.247730e-09 -0.126657  0.991947    2.2    5


Deferred evaluation
~~~~~~~~~~~~~~~~~~~

Expressions represent a computation that has not yet been executed, and will
be executed when we trigger it, for example by calling ``eval()`` or getting
the estimator and calling ``fit()``.

This means that we cannot use usual Python control flow statements such as
``if``, ``for``, ``with`` etc. with expressions, because those constructs would
execute immediately.

.. code:: python

    for column in orders.columns:
        pass
    # TypeError: This object is an expression that will be evaluated later, when your
    # pipeline runs. So it is not possible to eagerly iterate over it now.

We get an error because the ``for`` statement tries to iterate immediately
over the columns. However, ``orders.columns`` is not an actual list of
columns: it is a Skrub expression that will produce a list of columns, later,
when we run the computation.

This remains true even if we have provided a value for ``orders`` and we can
see a result for that value:

.. code:: python

    orders.columns
    # Index([item, price, qty])

So we must delay the execution of the ``for`` statement until the computation
actually runs and ``orders.columns`` has been evaluated.

We can do this by defining a function that contains the control flow statement we need,
and decorating it with ``skrub.deferred``. What this does is deferring
the calls to our function. Now when we call it, instead of running
immediately, it returns a skrub expression that wraps the function call. The
original function is actually called when we evaluate the expression.

.. code:: python

    @skrub.deferred
    def with_upper_columns(df):
        new_columns = [c.upper() for c in df.columns]
        return df.set_axis(new_columns, axis="columns")

    with_upper_columns(orders)
    # <Call 'with_upper_columns'>
    # Result:
    # ―――――――
    #    ITEM  PRICE  QTY
    # 0   pen    1.5    1
    # 1   cup    NaN    1
    # 2   pen    1.5    2
    # 3  fork    2.2    4

``skrub.deferred`` is useful for our own functions but also when we need to
call module-level functions from a library. For example to delay loading of a
csv file we could use something like:

.. code:: python

    csv_path = skrub.var("csv_path")
    data = skrub.deferred(pd.read_csv)(csv_path)

For the same reason (we are building a computation graph, not
immediately computing a single result), any transformation that we have must
not modify its input, but leave it unchanged and return a new value.

Think of the transformers in a scikit-learn pipeline: each computes a new
result without modifying its input.

.. code:: python

    orders['total'] = orders['price'] * orders['qty']
    # TypeError: Do not modify an expression in-place. Instead, use a function that
    # returns a new value.This is necessary to allow chaining several steps in a
    # sequence of transformations.
    # For example if df is a pandas DataFrame:
    # df = df.assign(new_col=...) instead of df['new_col'] = ...

Finally, there are other occasions where we may need to use ``deferred``:

- we have many nodes in our graph, and we want to collapse a sequence of
  steps into a single function call, that will appear as a single node
- we have steps that need to be deferred until the full computation runs,
  because they depend on the runtime environment or on objects that cannot be
  pickled together with the rest of the computation graph -- for example
  opening and reading a file.

.. rubric:: Examples

- See :ref:`example_expressions_intro` for an example of skrub pipelines using
  expressions on dataframes.
- See :ref:`example_tuning_pipelines` for an example of hyper-parameter tuning using
  skrub pipelines.
