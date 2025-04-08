# ruff: noqa: E501
"""
Skrub expressions
=================

Skrub provides special objects that represent intermediate results in a
computation. They record the different operations we perform on them (such as
applying operators or calling methods), which allows to later retrieve the
whole computation graph as a machine-learning estimator that can be fitted and
applied to unseen data.

Because those Skrub objects encapsulate a computation, which can be evaluated
to produce a result, we call them "expressions".

The simplest expressions are variables, which represent inputs to our
machine-learning estimator such as a ``"products"`` or a ``"customers"`` table.
Those variables can be combined with operators and function calls to build up
more complex expressions. The estimator is being constructed implicitly when we
apply those operations, rather than by providing an explicit list of
transformations.
"""

# %%
# We start by declaring inputs with ``skrub.var``.

# %%
import skrub

a = skrub.var("a")
b = skrub.var("b")

# %%
# We then apply transformations, composing more complex expressions.

# %%
c = a + b
c

# %%
# We can evaluate the expression.

# %%
c.skb.eval({"a": 10, "b": 6})

# %%
c.skb.eval({"a": 2, "b": 3})

# %%
# As shown above, the special ``.skb`` attribute provides access to skrub
# functionality to interact with the expression object itself. Access to any other
# attribute is simply added as a new operation in the computation
# graph:

# %%
d = c.capitalize()
d

# %%
d.skb.eval({"a": "hello, ", "b": "world!"})

# %%
# Finally, we can get an estimator that can be fitted and applied to data.

# %%
estimator = c.skb.get_estimator()
estimator.fit_transform({"a": 10, "b": 7})


# %%
# Handling multiple inputs
# ------------------------
#
# One advantage of the approach outlined above is that we can build models that
# handle and transform multiple tables (and other kinds of inputs).
#
# Scikit-learn pipelines represent a linear sequence of transformations on one
# table with a fixed number of rows.
#
# .. image:: ../../_static/sklearn_pipeline.svg
#     :width: 500
#
# Skrub expressions, on the other hand, can manipulate any number of variables.
# The transformation they perform is not a linear sequence but any Directed
# Acyclic Graph of computations.
#
# .. image:: ../../_static/skrub_expressions.svg


# %%
# Previews of the results
# -----------------------
#
# As we saw above, we can call ``eval()`` with a dictionary of bindings for the
# variables in our expression in order to compute the result of the pipeline.
# However, this is an operation that we want to do very often during development:
# to avoid having to call ``eval()`` repeatedly, skrub provides a way to
# preview the result of the expression as we work on it.
#
# When we create a variable, we can pass a value in addition to its name. If a value
# is provided, skrub will use it to compute the result of the expression on the
# value. This makes
# the development more interactive, allows catching errors early, and can
# provide better help and tab-completion in an interactive Python shell.
#
# Moreover, those values can be used to obtain a fitted estimator or
# cross-validation scores as we will see.
#
# Note that example values are immutable throughout the pipeline. This means that
# to change the value of a variable, we need to create the pipeline again with
# the new value.
#
# To provide an example value to ``var("a")``, we can write:

# %%
a = skrub.var("a", 10)  # note the value 10
b = skrub.var("b", 6)
c = a + b
c

# %%
# In the display above, you can still see the graph by clicking on the dropdown
# ``▶ Show graph``.
#
# Seeing results for the values we provided does *not* change the fact that we
# are building a computation graph that we want to reuse, not just computing
# the result for a fixed input. We can think of the displayed result as a
# preview of the output on one example dataset.

# %%
c.skb.eval({"a": 3, "b": 2})


# %%
# In what follows we will always provide values for our variables. When
# building your own estimator we recommend you do so as well, because running
# the estimator on example data every step of the way makes development much
# easier by catching errors as soon as the operation is added, rather than
# later when fitting the full estimator.

# %%
# Composing expressions
# ---------------------
#
# The simplest expressions are variables created with ``skrub.var``. Complex
# expressions are constructed by applying functions or operators to other
# expressions.
#
# So what operations are allowed? Remember that when we apply an operation to
# an expression, it is implicitly added as a step in our computation. When we
# run the computation, those operations are replayed on the actual data we
# provide.
#
# So we can use any operation that is valid for the types we will pass to run
# the computation (with some caveats detailed below). If we know we will pass a
# pandas DataFrame as the ``"products"`` table, we can use any of the methods
# of pandas DataFrames. If instead we know we will pass a numpy array, we use
# the methods of numpy arrays, etc.
#
# More operations (e.g. applying a scikit-learn estimator) are available
# through the special ``skb`` attribute, as we will see later.
#
# Basic operations
# ~~~~~~~~~~~~~~~~
#
# Suppose we want to process dataframes that look like this:

# %%
import pandas as pd

orders_df = pd.DataFrame(
    {
        "item": ["pen", "cup", "pen", "fork"],
        "price": [1.5, None, 1.5, 2.2],
        "qty": [1, 1, 2, 4],
    }
)
orders_df

# %%
# We can create a skrub variable to represent that input

# %%
orders = skrub.var("orders", orders_df)
orders

# %%
# Because we know we will feed a DataFrame to the computation, we manipulate
# ``products`` as if it were a DataFrame.
#
# We can access its attributes:

# %%
orders.columns

# %%
# Access the "items" column, indexing, slicing:

# %%
orders["item"].iloc[1:]

# %%
# Applying operators:

# %%
orders["price"] * orders["qty"]

# %%
# Calling methods:

# %%
orders.assign(total=orders["price"] * orders["qty"])

# %%
# It is important to note that the original ``orders`` pipeline is not modified
# by the previous cells. Instead, in each cell a new expression is created that
# represents the result of the operation.
#
# This is similar to how pandas and polars
# DataFrames work: when we call a method on a DataFrame, it returns a new
# DataFrame that represents the result of the operation, rather than modifying
# the original DataFrame in place. However, while in pandas it is possible to
# work on a DataFrame in place, skrub does not allow this.
#
# We can check this by looking at the graph of ``orders`` (which remains unmodified),
# and ``new_orders`` (which instead contains the new steps):
orders
# %%
new_orders = orders.assign(total=orders["price"] * orders["qty"])
new_orders

# %%
# Applying machine-learning estimators
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As mentioned above, in addition to those usual operations, the expressions
# have a special attribute: ``.skb``, which gives access to the methods and objects
# provided by skrub. A particularly important one is
# ``.skb.apply()``, which allows to scikit-learn estimators to the pipeline.

# %%
orders.skb.apply(skrub.TableVectorizer())

# %%
# It is also possible to apply a transformer to a subset of the columns:

# %%
orders.skb.apply(skrub.StringEncoder(n_components=3), cols="item")

# %%
# Again, the crucial part is that when applying such operations, the return
# value encapsulates the whole computation that produces the result we see. We are
# not interested in a single result for the example values we provided, but in
# the ability to retrieve a machine-learning estimator that we can fit and then
# apply to unseen data.

# %%
vectorized_orders = orders.assign(total=orders["price"] * orders["qty"]).skb.apply(
    skrub.TableVectorizer()
)
vectorized_orders

# We can use ``.skb.get_estimator(fitted=True)`` to retrieve the estimator and
# fit it on the data we provided.
# %%
estimator = vectorized_orders.skb.get_estimator(fitted=True)

# %%
# We can now use the fitted estimator to transform new data.
new_orders = pd.DataFrame({"item": ["fork"], "price": [2.2], "qty": [5]})
estimator.transform({"orders": new_orders})

# %%
# Deferred evaluation
# ------------------
#
# Expressions represent a computation that has not yet been executed, and will
# be executed when we trigger it, for example by calling ``eval()`` or getting
# the estimator and calling ``fit()``.
#
# This means that we cannot use usual Python control flow statements such as
# ``if``, ``for``, ``with`` etc. with expressions, because those constructs would
# execute immediately.

# %%
# For example, we cannot do this::
#
#     for column in orders.columns:
#         pass
#
# .. code-block:: none
#
#     TypeError: This object is an expression that will be evaluated later, when your pipeline runs. So it is not possible to eagerly iterate over it now.
#
# We get an error because the ``for`` statement tries to iterate immediately
# over the columns. However, ``orders.columns`` is not an actual list of
# columns: it is a skrub expression that will produce a list of columns, later,
# when we run the computation.
#
# .. image:: ../../_static/skrub_3d_2.svg
#   :width: 200
#
# This remains true even if we have provided a value for ``orders`` and we can
# see a result for that value:

# %%
orders.columns

# %%
# The "result" we see is an *example* result that the computation produces for
# the data we provided. But we want to fit our estimator and apply it many
# times to different datasets, for which it will return a new object every
# time. So even if we see a preview of the output on the data we provided,
# ``orders.columns`` still represents a future computation that remains to be
# evaluated.
#
# So we must delay the execution of the ``for`` statement until the computation
# actually runs and ``orders.columns`` has been evaluated.
#
# We can do this by wrapping it in a function and using ``skrub.deferred``.
# Suppose we want to convert the columns to upper case, and our first attempt
# runs into the problem we just described::
#
#     COLUMNS = [c.upper() for c in orders.columns]
#
# .. code-block:: none
#
#     TypeError: This object is an expression that will be evaluated later, when your pipeline runs. So it is not possible to eagerly iterate over it now.

# %%
# We define a function that contains the control flow statement we need. Then,
# we apply the ``skrub.deferred`` decorator to it. What this does is deferring
# the calls to our function. Now when we call it, instead of running
# immediately, it returns a skrub expression that wraps the function call. The
# original function is actually called when we evaluate the expression.


# %%
@skrub.deferred
def with_upper_columns(df):
    new_columns = [c.upper() for c in df.columns]
    return df.set_axis(new_columns, axis="columns")


with_upper_columns(orders)

# %%
# If you unfold the dropdown ``▶ Show graph`` above, you can
# see that a call to our function has been added to the computation graph.
#
# .. image:: ../../_static/skrub_3d_3.svg
#   :width: 200
#
# When the computation runs, ``orders`` will be evaluated first and the result (an
# actual dataframe) will be passed as the ``df`` argument to our function.
#
# .. image:: ../../_static/skrub_3d_4.svg
#   :width: 400

# %%
# ``skrub.deferred`` is useful for our own functions but also when we need to
# call module-level functions from a library. For example to delay loading of a
# csv file we could use something like:

# %%
csv_path = skrub.var("csv_path")
data = skrub.deferred(pd.read_csv)(csv_path)

# %%
# Also note that for the same reason (we are building a computation graph, not
# immediately computing a single result), any transformation that we have must
# not modify its input, but leave it unchanged and return a new value.
#
# Think of the transformers in a scikit-learn pipeline: each computes a new
# result without modifying its input.

# %%
# This would raise an error::
#
#     orders['total'] = orders['price'] * orders['qty']
#
# .. code-block:: none
#
#     TypeError: Do not modify an expression in-place. Instead, use a function that returns a new value.This is necessary to allow chaining several steps in a sequence of transformations.
#     For example if df is a pandas DataFrame:
#     df = df.assign(new_col=...) instead of df['new_col'] = ...
#
# In this case the error message suggests a solution: use ``assign()`` instead
# of item assignment. Luckily, we are mostly focussed on pandas and polars
# dataframes and they provide an API supporting this functional (returning
# transformed values without modifying the inputs) style for most operations
# (even more so polars than pandas).
#
# When we do need assignments or in-place transformations, we can put them in a
# ``deferred`` function. But remember to make a (shallow) copy and return the new value.
#


# %%
@skrub.deferred
def add_total(df):
    new_df = df.copy(deep=False)  # Creates a new dataframe but does not copy any data
    new_df["total"] = new_df["price"] * new_df["qty"]
    return new_df


add_total(orders)

# %%
# Finally, there are other occasions where we may need to use ``deferred``:
#
# - we have many nodes in our graph, and we want to collapse a sequence of
#   steps into a single function call, that will appear as a single node
# - we have steps that need to be deferred until the full computation runs,
#   because they depend on the runtime environment or on objects that cannot be
#   pickled together with the rest of the computation graph -- for example
#   opening and reading a file.
