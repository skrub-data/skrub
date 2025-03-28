"""
Skrub expressions
=================

We saw in the previous example how skrub expressions can help you to apply estimators
and evaluate data pipelines with ease. We will now take a closer look at
the underlying mechanism of expressions.

A refresher on sklearn pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Why are sklearn pipelines useful in practice? By chaining multiple transformers
and estimators together in a single trainable object, they empower you to:

- Evaluate your model with more confidence by using cross-validation
- Search for the best hyper-parameters across all your estimators.
- Serialize all estimators at once, which eases versioning and deployment as you
  only have one model to save and load.

.. image:: ../../_static/sklearn_pipeline.svg
    :width: 500

Skrub expressions can build a pipeline on multiple tables, akin to an execution graph
or a DAG. This generalization allows a greater flexibility and extends
cross-validation to the earliest stages of data wrangling. Skrub expressions also
ease the definition of simple pipelines and of their hyper-parameter spaces.

.. image:: ../../_static/skrub_expressions.svg

This schema is based on the baskets fraud use-case we saw in the previous example.
Our goal is to find the best hyper-parameters for our transformer and estimator.

As often with real-world modeling, we have to derive the design matrix ``X`` and the
target ``y`` from multiple tables. If we mark which steps results in ``X`` or ``y``,
skrub expressions are able to cross-validate all subsequent steps (including
"transformer 1" and "estimator") and search their best user-defined hyper-parameters.

You can output a fully fledge estimator from this compute graph, or a
grid search/randomized search cv.

In the sections below, we explore each part of this schema.

Skrub expressions are lazy
~~~~~~~~~~~~~~~~~~~~~~~~~~

When you define an operation on a skrub variable, skrub expressions build an execution
graph in the background to be used on new data. This mechanism is similar to the
execution planner built automatically in lazy mode with Spark or Polars for example.

Let's see this on a simple example.
"""
# %%
# Skrub variables are the inputs of your graph, and naming them is mandatory to run
# this graph later on new data.
# Note that **giving a value to a variable is optional** and will allow you to have
# a preview on the operation and develop interactively.
import skrub

# Without value
loans = skrub.var("loans")
loans

# %%
# Let's set a value to make this example more interactive.
import pandas as pd

loans_df = pd.DataFrame(
    {
        "loan_id": [1, 2, 3, 4],
        "amount_requested": [500, 200, 30, 150],
        "type": ["A", "B", "A", "C"],
        "is_accepted": [False, True, True, False],
    }
)
loans = skrub.var("loans", value=loans_df)
loans

# %%
# Skrub variables can represent any type.
currency_rate = skrub.var("currency_rate", 1.3)
currency_rate

# %%
# Skrub variables allow you to use the API of their underlying objects directly.
# ``loans`` have the same methods and attributes as a regular Pandas dataframe, but
# due to the lazy nature of our expressions, **inplace operations are impossible**.
#
# .. code-block:: python
#
#   # This will fail
#   loans["total_requested"] = loans["amount_requested"] * currency_rate

# This will work fine
loans = loans.assign(total_requested=loans["amount_requested"] * currency_rate)
loans

# %%
# You can easily inspect the graph by clicking on ``▶ Show graph`` or by running:
loans.skb.draw_graph()

# %%
# .. note::
#
#   The namespace ``skb`` allows you to access expression methods to interact with
#   the graph.
#
# To run the graph and obtain the transformed dataframe on any new input data,
# ``skb.eval`` is the equivalent of running ``fit_transform`` on a regular sklearn
# pipeline. ``environment`` denotes the set of input to be used:
transformed_df = loans.skb.eval(environment={"loans": loans_df, "currency_rate": 2.0})
transformed_df

# %%
# Applying estimators to expressions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# At this stage, you may wonder why you should be using skrub expressions instead of a
# plain function. Let's add transformers and estimators with ``skb.apply`` to make this
# graph stateful!
#
# To apply a learner, we need to explicitly tell the graph which
# step corresponds to our design matrix ``X`` and target ``y``, using
# ``skb.mark_as_x()`` and ``skb.mark_as_y()`` respectively.
#
# We also inline a search space for the C hyper-parameter of the logistic regression
# using ``skrub.choose_float()``.
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

X = loans[["total_requested", "type"]].skb.mark_as_X()
y = loans["is_accepted"].skb.mark_as_y()

loans_preprocessed = X.skb.apply(
    OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist"),
    cols="type",  # Only transform this column
)
prediction = loans_preprocessed.skb.apply(
    LogisticRegression(C=skrub.choose_float(1e-5, 10, log=True, name="C")),
    y=y,  # This is where we set the target!
)
prediction

# %%
# Remember that the ``prediction`` output is **only a preview of the results**.
# This means that although we ran each step to check the correctness of the expressions,
# we didn't return a fitted object. Now that we have completed this graph, we can turn
# it into an estimator and fit it, before serializing it.

estimator = prediction.skb.get_estimator()
estimator.fit(environment={"loans": loans_df, "currency_rate": 2.0})
print(estimator.predict_proba({"loans": loans_df.head(1), "currency_rate": 2.0}))

# %%
# .. code-block:: python
#
#   import pickle
#
#   with open("my_estimator.pkl", "wb") as f:
#       pickle.dump(estimator, f)
#
# Also notice how ``X`` and ``y`` are identified in the graph.
prediction.skb.draw_graph()

# %%
# .. warning::
#   Hyper-parameter tuning can only be executed on nodes defined **downstream**
#   to ``X``, because folds are set using ``X`` indices and propagated to the rest
#   of the graph.
#
# Here, we would only be able to search hyper-parameters in the ``OneHotEncoder``
# and ``LogisticRegression`` steps, but not in operations before marking ``X``.
#
# We can cross-validate and run hyper-parameter tuning via:

randomized_search_cv = prediction.skb.get_randomized_search(cv=2, n_iter=2, fitted=True)
randomized_search_cv.results_

# %%
# We can even run nested cross-validation on grid-search to avoid getting
# optimistic performances.
from skrub._expressions._estimator import cross_validate

cross_validate(
    randomized_search_cv,
    environment={"loans": loans_df.sample(100, replace=True), "currency_rate": 2.0},
    cv=2,
)

# %%
# Control flow using your data values
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Since we are building a lazy chain of operations, **we cannot use variable data
# values for control flows like if-else conditions or for-loops directly**.
#
# What does this means in practice?
#
# .. code-block:: python
#
#  # This will fail
#  for col in loans.columns:
#      print(loans[col].max())

# But this will work:
print(loans.max(axis=0))


# %%
# To understand this concept, think of adding steps to a data processing pipeline.
# Skrub variables without assigned values act as placeholders. They delay actual
# computation until real values are provided.
#
# .. image:: ../../_static/skrub_3d_1.svg
#   :width: 300
#
# |
#
# You can define pipeline steps only if they are not conditional on specific data
# values. This is because the data might change each time the pipeline runs.
# For example, the previous for-loop running on the dataframe columns
# fails because skrub expressions don't contain actual data; they merely describe a
# sequence of operations.
#
# |
#
# .. image:: ../../_static/skrub_3d_2.svg
#   :width: 300
#
# |
#
# Instead, we need to bring this for-loop logic into a ``print_max`` function, and
# decorate it using ``@skrub.deferred``. This decoration turns ``print_max`` into
# a step in the pipeline graph. The function then gets actual data values only when
# the pipeline runs.
#
# .. image:: ../../_static/skrub_3d_3.svg
#   :width: 300
#
@skrub.deferred
def print_max(loans):
    for col in loans.columns:
        print(loans[col].max())
    return loans


# %%
# |
#
# If you instantiate skrub variables with values, the pipeline executes immediately
# after each step. This approach combines lazy pipeline creation with eager evaluation,
# allowing you to preview results and work interactively.
# Without assigning values, the pipeline runs only when explicitly executed
# with ``skb.eval()`` or when fitting models or cross-validation.
#
# |
#
# .. image:: ../../_static/skrub_3d_4.svg
#   :width: 600

loans = print_max(loans)

# %%
# Indeed, you can see that the ``print_max`` function has been added as a step.
loans.skb.draw_graph()

# %%
# The next example will focus on the hyper-parameters tuning capabilities of skrub
# expressions.
