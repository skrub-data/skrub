# ruff: noqa: E501
"""
Skrub expressions
=================

The way we build a machine-learning pipeline in skrub is somewhat different
from scikit-learn. We do not create the complete pipeline ourselves by
providing an explicit list of transformation steps. Rather, we manipulate
objects that represent intermediate results in our computation. They record the
different operations we perform on them (such as applying operators or calling
methods), which allows to later retrieve the whole sequence of operations as a
scikit-learn estimator that can be fitted and applied to unseen data.

Because those objects encapsulate a computation, which can be evaluated to
produce a result, we refer to them as "expressions".

The simplest expressions are variables, which represent inputs to our pipeline
such as a ``"products"`` or a ``"customers"`` table. Those can be combined with
operators and function calls to build up more complex expressions. The pipeline
is being built implicitly when we apply those operations.

Working with Skrub expressions is similar to working on a pandas or polars
DataFrame: at each step, a new expression object that represents an operation
is created, in the same vein that a new DataFrame is created when we apply a
method to a DataFrame. The difference is that the expression object does not
contain the result of the operation, but rather a description of the operation to
be performed: expressions are always deferred (similar to LazyFrame in polars).

We had a quick preview of skrub expressions in the previous example.
In this one we explain how they work in detail.
The next example will cover hyperparameter search.

"""

# %%
# General workflow
# ----------------
#
# Skrub expressions exist to build machine-learning pipelines that contain
# scikit-learn estimators. However, they can process any kind of data.
# To start with a very simple example and give a summary of the different steps
# let us build a pipeline that just adds 2 numbers.


# %%
# Declare inputs with ``skrub.var``

# %%
import skrub

a = skrub.var("a")
b = skrub.var("b")

# %%
# Apply transformations, composing more complex expressions

# %%
c = a + b
c

# %%
# The expression ``c`` is a skrub expression that represents the result of
# adding ``a`` and ``b``. It is not the result of the addition, but a
# description of the operation to be performed.
# In the display above, you can see the diagram representing the pipeline
# by clicking on the dropdown ``▶ Show graph``.
# We can then evaluate the expression (``eval()`` is not very often used in
# practice but we use it in the example for explaining how expressions work.):

# %%
c.skb.eval({"a": 10, "b": 6})

# %%
c.skb.eval({"a": 2, "b": 3})

# %%
# After building the expression, it is possible to retrieve an estimator that can be
# fitted and applied to data: this is done with ``.skb.get_estimator()``

# %%
estimator = c.skb.get_estimator()
estimator.fit_transform({"a": 10, "b": 7})

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
# By providing a value that is a realistic example of the data we will use, we can
# have a clear idea of how the pipeline will behave when we run it on real data.
#
# Moreover, providing example values for the variables allows to obtain a fitted
# pipeline and cross-validation scores, as we will see in a later example.
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
# It is important to understand that the result we see here is simply a "preview"
# of the result of the expression, computed with the values we provided. Internally,
# skrub is building a pipeline that adds two values, and providing different values
# will produce the same pipeline but with different results:

# %%
c.skb.eval({"a": 3, "b": 2})


# %%
# In what follows we will always provide values for our variables. We recommend
# you do so when you are building your own pipeline as well:  running
# the pipeline on example data every step of the way makes development much
# easier by catching errors as soon as the operation is added, rather than
# later when the pipeline is fitted.

# %%
# Composing expressions
# ---------------------
#
# The simplest expressions are variables created with ``skrub.var`` (or plain
# python objects). Complex expressions are constructed by applying functions or
# operators to other expressions.
#
# So, what operations are allowed? Remember that when we apply an operation to
# an expression, it is implicitly added as a step in our pipeline. When we call
# ``.fit()`` (or ``.transform()``, ``.predict()`` etc) on the pipeline, those
# operations are replayed on the actual data provided to ``fit()``.
#
# We can use any operation that is valid for the data we will pass to
# ``fit()`` (with some caveats detailed below). If we know we will pass a
# pandas DataFrame as the ``"products"`` table, we can use any of the methods
# of pandas DataFrames. If instead we know we will pass a numpy array, we use
# the methods of numpy arrays.
#
# More operations (e.g. applying a scikit-learn estimator) are available
# through the special ``skb`` attribute, as we will see later.
#
# Basic operations
# ~~~~~~~~~~~~~~~~
#
# Suppose we want our pipeline to process dataframes that look like this:

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
# We can create a skrub variable to represent that input to the pipeline:

# %%
orders = skrub.var("orders", orders_df)
orders

# %%
# Because we know we will feed a DataFrame to the pipeline, we manipulate
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
# value encapsulates the whole pipeline that produces the result we see, and the
# previous step remains unmodified.
# This allows to backtrack easily to previous steps in the pipeline, provided
# that the previous step has not been modified.
#
# By encapsulating the pipeline and providing a standalone object we can retrieve
# a machine-learning estimator that can be fitted and applied to unseen data.


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
# Delayed evaluation
# ------------------
#
# It is important to understand that expressions represent a computation that
# has not yet been executed, and will be executed when we trigger it, for
# example by calling ``eval()`` or getting the estimator and calling ``fit()``.
#
# This means that we cannot use usual Python control flow statements with
# expressions, because those would execute immediately.

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
# when we run the pipeline.
#
# This remains true even if we have provided a value for ``orders`` and we can
# see a result for that value:

# %%
orders.columns

# %%
# The "result" we see is an *example* result that the pipeline produces for the
# data we provided. However, since we want to fit our pipeline and apply it many
# times to different datasets, it will return a new object every time. This means
# that, even if we see a preview of the pipeline's output on the data we provided,
# ``orders.columns`` still represents a future computation that remains to be
# evaluated.
#
# For this reason we must delay the execution of the ``for`` statement until the
# pipeline actually runs, and ``orders.columns`` has been evaluated.
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
# original function is only called when we evaluate the expression, by
# running the pipeline.


# %%
@skrub.deferred
def with_upper_columns(df):
    new_columns = [c.upper() for c in df.columns]
    return df.set_axis(new_columns, axis="columns")


with_upper_columns(orders)

# %%
# If you unfold the dropdown ``▶ Show graph`` above, you can
# see that a call to our function has been added to the pipeline.
#
# When the pipeline runs, ``orders`` will be evaluated first and the result (an
# actual dataframe) will be passed as the ``df`` argument to our function.

# %%
# Here is another example of using ``skrub.deferred``.


# %%
def check(price):
    if price < 0:
        print("warning! negative price:", price)
    else:
        print("good price:", price)
    return price


check(2.5)

# %%
# We cannot use ``check`` directly on an expression:

# %%
price = skrub.var("price")
price

# %%
price < 0

# %%
# .. code-block::
#
#     check(price)
#
# .. code-block:: none
#
#     TypeError: This object is an expression that will be evaluated later, when your pipeline runs. So it is not possible to eagerly use its Boolean value now.
#
# It is the same kind of error: the ``if`` statement tries to convert
# ``price < 0`` to a Python Boolean but it is a skrub expression that needs to be
# evaluated first.


# %%
# Instead, if we defer the execution of ``check`` we get an function that adds
# the call to the pipeline and returns an expression:

# %%
expr = skrub.deferred(check)(price)
expr

# %%
# We can then evaluate the expression and obtain the result of the check:
expr.skb.eval({"price": 2.5})

# %%
expr.skb.eval({"price": -3.0})

# %%
# Remember that providing a value when the variable ``"price"`` is initialized
# does *not* change the nature of the expression; even if it provides a preview,
# it is still a computation that can be evaluated many times on different inputs.

# %%
price = skrub.var("price", -3.0)
price
# %%
# We still cannot call ``check(price)`` directly, for the same reasons.
# If ``check`` ran immediately, we would be checking our example price -3.0,
# rather than adding the check to our pipeline so that it is applied to all
# inputs that we feed through it. So we must still use ``deferred``. The
# difference, now that we provided a value, is that besides creating an
# expression and returning it, skrub also immediately evaluates it on the
# example data and *shows* the result.

# %%
skrub.deferred(check)(price)

# %%
# ``skrub.deferred`` is useful for our own functions, but also when we need to
# call module-level functions from a library. Suppose we want to add a binary
# feature indicating missing values in the ``"price"`` column. We could do this
# with the Series' method ``orders['price'].isna()``, but let us imagine it does
# not exist or we prefer to use the module-level ``pd.isna``. Calling the function
# directly on our expression raises an error::
#
#     pd.isna(orders['price'])
#
# .. code-block:: none
#
#     TypeError: This object is an expression that will be evaluated later, when your pipeline runs. So it is not possible to eagerly use its Boolean value now.
#
# This is the same error as above: somewhere inside of ``isna``, there is an
# ``if`` statement that errors because we passed an expression rather than a
# Series. Here again the solution is to use ``deferred``:

# %%
null_price = skrub.deferred(pd.isna)(orders["price"])
null_price

# %%
orders.assign(missing_price=skrub.deferred(pd.isna)(orders["price"]))


# %%
# Also note that for the same reason (we are building a pipeline, rather than
# computing immediately a single result), any transformation that we have must
# leave its input it unchanged and return a new value.
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
# - We have many nodes in our graph, and we want to collapse a sequence of
#   steps into a single function call to convert it to a single node
# - We have steps that need to be deferred until the pipeline runs, because
#   they depend on the runtime environment, or on objects that cannot be pickled
#   together with the rest of the pipeline -- for example opening and reading a
#   file.
#
# This concludes the introduction to skrub expressions. We have seen how to create
# an expression, assign example values to it, and apply transformations. We also
# discussed how execution of functions is deferred.
# In the next example, we will show how to apply the expressions to a scikit-learn
# pipeline.
