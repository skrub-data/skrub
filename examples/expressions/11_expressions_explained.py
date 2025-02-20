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
is being built implicitly when we apply those operations

We had a quick preview of skrub expressions in the previous example.
In this one we explain how they work in detail.
The next example will cover hyperparameter search.

"""

# %%
# General workflow
# ----------------
#
# Skrub expressions exist to build machine-learning pipelines that contain
# scikit-learn estimators. However note that they can process any kind of data.
# To start with a very simple example and give a summary of the different steps
# let us build a pipeline that adds 2 numbers.


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
# We can evaluate the expression. (``eval()`` is not very often used in
# practice but we use it in the example for explaining how expressions work.)

# %%
c.skb.eval({"a": 10, "b": 6})

# %%
c.skb.eval({"a": 2, "b": 3})

# %%
# Get a scikit-learn estimator that can be fitted and applied to data

# %%
estimator = c.skb.get_estimator()
estimator.fit_transform({"a": 10, "b": 7})

# %%
# Previews of the results
# -----------------------
#
# As we saw above, we can call ``eval()`` with a dictionary of bindings for the
# variables in our expression in order to compute the result. However, seeing
# the result of what we have built so far is something we want to do very
# often. So skrub helps us avoid needing to call ``eval()`` repeatedly.
#
# When creating a variable, we can pass a value in addition to its name. When
# those values are available, whenever we create a new expression, skrub
# computes the result on the provided values and can show it to us. This makes
# the development more interactive, allows catching errors early, and can
# provide better help and tab-completion in an interactive Python shell.
#
# Moreover, those values can be used to obtain a fitted pipeline or
# cross-validation scores as we will see.
#
# Instead of simply ``var("a")`` as before, we can write:

# %%
a = skrub.var("a", 10)  # note the value 10
b = skrub.var("b", 6)
c = a + b
c

# %%
# It is important to understand that seeing results for the values we provided
# does *not* change the fact that we are building a pipeline that we want to
# reuse, not just computing the result for a fixed input. Think of the
# displayed result as a preview of the pipeline's output on one example
# dataset. Providing values does not otherwise change the behavior of the
# expressions.

# %%
c.skb.eval({"a": 3, "b": 2})


# %%
# In what follows we will always provide values for our variables. When
# building your own pipeline we recommend you do so as well, because running
# the pipeline on example data every step of the way makes development much
# easier by catching errors as soon as the operation is added, rather than
# later when fitting the pipeline.

# %%
# Composing expressions
# ---------------------
#
# The simplest expressions are variables created with ``skrub.var`` (or plain
# python objects). Complex expressions are constructed by applying functions or
# operators to other expressions.
#
# So what operations are allowed? Remember that when we apply an operation to
# an expression, it is implicitly added as a step in our pipeline. When we call
# ``.fit()`` (or ``.transform()``, ``.predict()`` etc) on the pipeline, those
# operations are replayed on the actual data provided to ``fit()``.
#
# So we can use any operation that is valid for the data we will pass to
# ``fit()`` (with some caveats detailed below). If we know we will pass a
# pandas DataFrame as the ``"products"`` table, we can use any of the methods
# of pandas DataFrames. If instead we know we will pass a numpy array, we use
# the methods of numpy arrays.
#
# More operations (e.g. applying a scikit-learn estimator) are available
# through the special ``skb`` attribute, as we will see later.
#
# Suppose we want our pipeline to process dataframes that look like this:

# %%
import pandas as pd

orders_df = pd.DataFrame(
    {
        "item": ["pen", "cup", "pen", "fork"],
        "price": [1.5, None, 1.5, 2.1],
        "qty": [1, 1, 2, 3],
    }
)

# %%
# We can create a skrub variable to represent that input to the pipeline

# %%
orders = skrub.var("orders", orders_df)
orders

# %%
# Because we know we will feed a DataFrame to the pipeline, we manipulate
# ``products`` as if it were a DataFrame.
#
# Accessing attributes:

# %%
orders.columns

# %%
# Accessing items, indexing, slicing:

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
# As mentioned above, in addition to those usual operations, the expressions
# have an additional special attribute: ``.skb`` gives access to the
# functionality provided by skrub. A particularly important one is
# ``.skb.apply()`` which allows applying scikit-learn estimators.

# %%
orders.skb.apply(skrub.TableVectorizer())

# %%
# It is also possible to apply a transformer to only some columns:

# %%
orders.skb.apply(skrub.StringEncoder(n_components=3), cols="item")

# %%
# Again, the crucial part is that when applying such operations, the return
# value encapsulates the whole pipeline that produces the result we see. We are
# not interested in a single result for the example values we provided, but in
# the ability to retrieve a machine-learning estimator that we can fit and then
# apply to unseen data.

# %%
vectorized_orders = orders.assign(total=orders["price"] * orders["qty"]).skb.apply(
    skrub.TableVectorizer()
)
vectorized_orders

# %%
estimator = vectorized_orders.skb.get_estimator(fitted=True)

# %%
new_orders = pd.DataFrame({"item": ["fork"], "price": [2.1], "qty": [7]})
estimator.transform({"orders": new_orders})

# %%
# What if we want to apply a function to our expression?
# Suppose we want to use the pandas module-level ``isna``
# The following raises an error::
#
#     pd.isna(orders['price'])
#
# Produces:
#
# ``TypeError: This object is an expression that will be evaluated later, when your pipeline runs. So it is not possible to eagerly use its Boolean value now.`` # noqa: E501

# %%
# Indeed ``orders['price']`` is not a pandas Series, it is a skrub expression that will
# produce a pandas Series when evaluated. So we need to defer the call to
# ``pd.isna`` until we have an actual argument to give it.
#
# (Of course here an alternative would be to use the Series' `.isna()` method)

# %%
null_price = skrub.deferred(pd.isna)(orders["price"])
null_price

# %%
# Delayed evaluation
# ------------------
#
# What we saw above is important to understand: expressions represent a
# computation that has not yet been executed, and will be executed when we
# trigger it, for example by calling ``eval()`` or getting the estimator and
# calling ``fit()``.
#
# This means that we cannot use usual Python control flow statements with
# expressions, because those would execute immediately.

# %%
# We cannot do this::
#
#     for column in orders.columns:
#         pass
#
# This would result in:
#
# ``TypeError: This object is an expression that will be evaluated later, when your pipeline runs. So it is not possible to eagerly iterate over it now.`` # noqa: E501
#

# %%
# The solution is to use ``skrub.deferred``
#
# bad::
#
#     COLUMNS = [c.upper() for c in orders.columns]
#
# This would result in:
#
# ``TypeError: This object is an expression that will be evaluated later, when your pipeline runs. So it is not possible to eagerly iterate over it now.`` # noqa: E501

# %%
# good:


# %%
@skrub.deferred
def with_upper_columns(df):
    new_columns = [c.upper() for c in df.columns]
    return df.set_axis(new_columns, axis="columns")


with_upper_columns(orders)

# %%
# Also note that for the same reason (we are building a pipeline, not
# immediately computing a single result), any transformation that we have must
# not modify its input, but leave it unchanged and return a new value.

# %%
# This would raise an error::
#
#     orders['total'] = orders['price'] * orders['qty']
#
# prints::
#
#     TypeError: Do not modify an expression in-place. Instead, use a function that returns a new value.This is necessary to allow chaining several steps in a sequence of transformations. # noqa
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


# %%
@skrub.deferred
def add_total(df):
    new_df = df.copy(deep=False)  # Creates a new dataframe but does not copy any data
    new_df["total"] = new_df["price"] * new_df["qty"]
    return new_df


add_total(orders)


# %%
# - show `.skb.full_report()`
#
# Cross-validation
# ----------------
#
# - explain ``.skb.mark_as_x()``, ``.skb.mark_as_y()``, ``skrub.X()``, ``skrub.y()``
# - show ``.skb.cross_validate()``
# - show ``.skb.get_estimator()`` again
#
# Conclusion
# ----------
#
# There is more: see the next example for hyperparameter search.
#
# A few more advanced features have not been shown and remain for more
# specialized examples, for example:
#
# - naming nodes, passing the value for any node with the inputs
# - ``.skb.applied_estimator``
# - ``.skb.concat_horizontal``, ``.skb.drop`` and ``.skb.select``, skrub selectors
# - ``.skb.freeze_after_fit`` (niche / very advanced)
