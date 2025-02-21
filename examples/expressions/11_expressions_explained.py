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
# In the display above, you can still see the diagram representing the pipeline
# by clicking on the dropdown ``▶ <BinOp: add>``.
#
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
# Applying machine-learning estimators
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As mentioned above, in addition to those usual operations, the expressions
# have a special attribute: ``.skb`` gives access to the
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
# data we provided. But we want to fit our pipeline and apply it many times to
# different datasets, for which it will return a new object every time. So even
# if we see a preview of the pipeline's output on the data we provided,
# ``orders.columns`` still represents a future computation that remains to be
# evaluated.
#
# So we must delay the execution of the ``for`` statement until the pipeline
# actually runs and ``orders.columns`` has been evaluated.
#
# We can do this by wrapping it in a function and using ``skrub.deferred``
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
# we apply the ``skrub.deferred`` decorator to it. What this does is to defer
# the calls to our function. Now when we call it, instead of running
# immediately, it returns a skrub expression that wraps the function call. The
# original function is actually called when we evaluate the expression, by
# running the pipeline.


# %%
@skrub.deferred
def with_upper_columns(df):
    new_columns = [c.upper() for c in df.columns]
    return df.set_axis(new_columns, axis="columns")


with_upper_columns(orders)

# %%
# If you unfold the dropdown ``▶ <Call 'with_upper_columns'>`` above, you can
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
# We could not use ``check`` directly on an expression:

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
# If we defer the execution of ``check`` we get an function that adds the call to
# the pipeline and returns expression instead:

# %%
expr = skrub.deferred(check)(price)
expr

# %%
expr.skb.eval({"price": 2.5})

# %%
expr.skb.eval({"price": -3.0})

# %%
# Providing a value when we initialize the variable ``"price"`` does *not*
# change the nature of the expression; even if it provides a preview it is
# still a computation that can be evaluated many times on different inputs.

# %%
price = skrub.var("price", -3.0)

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
# ``skrub.deferred`` is useful for our own functions but also when we need to
# call module-level functions from a library. Suppose we want to add a binary
# feature indicating missing values in the ``"price"`` column. We could do this
# with the Series' method ``orders['price'].isna()`` but let us imagine it does
# not exist or we prefer to use the module-level ``pd.isna``. Calling it
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
# Series. Here again the solution is to use ``deferred``.

# %%
null_price = skrub.deferred(pd.isna)(orders["price"])
null_price

# %%
orders.assign(missing_price=skrub.deferred(pd.isna)(orders["price"]))


# %%
# Also note that for the same reason (we are building a pipeline, not
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
# Finally, other occasions where we may need to use ``deferred`` are:
#
# - we have many nodes in our graph, and we want to collapse a sequence of
#   steps into a single function call, that will appear as a single node
# - we have steps that need to be deferred until the pipeline runs, because
#   they depend on the runtime environment or on objects that cannot be pickled
#   together with the rest of the pipeline -- for example opening and reading a
#   file.
#

# %%
# Supervised estimators and cross-validation
# ------------------------------------------
#
# We can use ``.skb.apply()`` to add scikit-learn transformers to the pipeline,
# but also to add a supervised learner like ``HistGradientBoostingClassifier``
# or ``LogisticRegression``.
#
# When we have built a full pipeline, we want to run cross-validation to
# measure its performance. We may also want to tune hyperparameters (shown in
# the next example).
#
# Skrub can run the cross-validation for us. In order to do so, it needs to
# split the dataset into training and testing sets. For this to happen, we need
# to tell skrub which items in our pipeline constitute the feature matrix ``X``
# and the targets (labels, outputs) ``y``.
#
# Indeed, a skrub pipeline can accept inputs that are not yet neatly organized
# into correctly-aligned ``X`` and ``y`` matrices. There may be some steps
# (such as loading data, performing joins or aggregations, separating features
# and targets from a single source) that need to be performed in order to
# construct the design matrix the targets. This makes skrub pipelines flexible
# and powerful. In this setting, in order to be able to split the data and
# perform cross-validation, we need to explicitly mark the intermediate results
# that should be considered as ``X`` and ``y`` and used to define
# cross-validation splits.
#
# In the simplest case where we have ``X`` and ``y`` available right from the
# start, we can indicate that simply by creating the input variables with
# ``skrub.X()`` and ``skrub.y()`` instead of ``skrub.var()``.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

import skrub.datasets

dataset = skrub.datasets.fetch_employee_salaries()

employees = skrub.X(dataset.X)
salaries = skrub.y(dataset.y)

vectorizer = skrub.TableVectorizer(
    high_cardinality=skrub.MinHashEncoder(n_components=8)
)
predictions = employees.skb.apply(vectorizer).skb.apply(
    HistGradientBoostingRegressor(), y=salaries
)
predictions

# %%
# Note how ``apply`` works for supervised estimators: we pass the targets
# (another skrub expression) as the argument for the parameter ``y``.
#
# We can now use another method from the special ``.skb`` attribute:
# ``cross_validate()``. It accepts the same parameters (except for ``X``, ``y``
# and ``estimator``) and returns the same output as
# ``sklearn.model_selection.cross_validate``.

# %%
predictions.skb.cross_validate()

# %%
# When the situation is more complex and we need some data processing to
# construct ``X`` and ``y``, we can still do that as part of the skrub pipeline
# and keep those transformations neatly bundled with the actual learning. We
# just need to tell skrub which intermediate results constitute ``X`` and
# ``y``. For example, suppose we get all the data as a single dataframe and we
# need to separate the target column from the rest. This is a very simple, toy
# example but more complex situations are handled in the same way.
#
# To indicate which intermediate results to split, we call ``.skb.mark_as_x()``
# and ``.skb.mark_as_y()`` on the appropriate objects:

# %%
full_data = skrub.var("data", dataset.employee_salaries)

employees = full_data.drop(columns="current_annual_salary").skb.mark_as_x()
salaries = full_data["current_annual_salary"].skb.mark_as_y()

vectorizer = skrub.TableVectorizer(
    high_cardinality=skrub.MinHashEncoder(n_components=8)
)
predictions = employees.skb.apply(vectorizer).skb.apply(
    HistGradientBoostingRegressor(), y=salaries
)
predictions

# %%
# If you unfold the dropdown to see the computation graph, you will see the
# inner nodes that have been marked as ``X`` and ``y``. To evaluate the
# pipeline, skrub first runs all the prior steps until it has computed ``X``
# and ``y``. Then, it splits those (with the usual scikit-learn
# cross-validation tools, as is done in ``sklearn.cross_validate``) and runs
# the rest of the pipeline inside of the cross-validation loop.

# %%
predictions.skb.cross_validate()

# %%
# Start your pipeline by constructing ``X`` and ``y``.
# Naturally, the construction of ``X`` and ``y`` has to run before splitting
# into cross-validation folds; it happens outside of the cross-validation loop.
#
# This means that any step we perform in this part of the pipeline has access
# to the full data (training and test sets). We should not use estimators that
# learn from the data before reaching the cross-validation loop, or we might
# obtain optimistic scores due to data leakage. Moreover, no choices or
# hyperparameters can be tuned in this part of the pipeline (tuning is
# discussed in more detail in the next example).
#
# Therefore, we should build ``X`` and ``y`` at the very start of the pipeline
# and use ``mark_as_x()`` and ``mark_as_y()`` as soon as possible -- as soon as
# we have separate ``X`` and ``y`` tables that are aligned and have one row per
# sample. In particular we should use ``mark_as_x()`` before doing any feature
# extraction and selection.
#
# Coming back to the previous example we could have written:

# %%
full_data = skrub.var("data", dataset.employee_salaries)

employees = full_data.drop(columns="current_annual_salary")
salaries = full_data["current_annual_salary"].skb.mark_as_y()

vectorizer = skrub.TableVectorizer(
    high_cardinality=skrub.MinHashEncoder(n_components=8)
)
features = employees.skb.apply(vectorizer).skb.mark_as_x()
predictions = features.skb.apply(HistGradientBoostingRegressor(), y=salaries)

predictions

# %%
# Note that ``.mark_as_x()`` has been moved down, until after the vectorizer is
# applied. Our pipeline can still run, but cross-validation now works
# differently: first the whole table is vectorized, _then_ the resulting
# feature matrix (and the targets) is split into cross-validation folds. Thus
# the vectorizer sees the full dataset. It is not much of an issue with the
# ``MinHashEncoder`` which does not learn anything, but if we used for example
# the ``GapEncoder`` it would use the testing data to learn its latent
# representations and might cause some overfitting. If we had some supervised
# transformations (that use the targets) such as feature selection with
# ``SelectKBest()`` before ``mark_as_x()``, the overfitting could be severe.
#
# Skrub will still let us apply transformers before reaching ``mark_as_x``
# because sometimes we know our transformer is stateless (eg the
# ``MinHashEncoder``, or the ``TextEncoder`` with ``n_components=None``) and it
# is faster to apply it once to the whole dataset than to recompute the
# transformation inside each cross-validation fold. But in general we should be
# careful and remember to separate features and targets and then use
# ``mark_as_x()`` as soon as possible.


# %%
# A full pipeline
# ---------------
#
# Now we know enough about skrub expressions to create a full pipeline for a
# realistic example and evaluate its performance.
#
# We finally come back to the credit fraud dataset from the previous example.

# %%
import skrub.datasets

dataset = skrub.datasets.fetch_credit_fraud()

# %%
# The ``baskets`` are orders on a e-commerce website

# %%
baskets = skrub.var("baskets", dataset.baskets[["ID"]]).skb.mark_as_x()
baskets

# %%
# Each is associated with a fraud flag

# %%
fraud_flags = skrub.var("fraud_flags", dataset.baskets["fraud_flag"]).skb.mark_as_y()
fraud_flags

# %%
# And each basket contains one or several products. The ``"basket_ID"`` column
# in the ``"products"`` refers to the ``"ID"`` column in the ``"baskets"``
# table. Given the information we have about a basket, we want to predict if
# its purchase was fraudulent.

# %%
products = skrub.var("products", dataset.products)
products

# %%
# Just out of curiosity, we can check the products for a few fraudulent
# baskets, to see if we can spot whether they have something in common.
#
# Note this is just for exploration, we will not add those computations to our
# prediction pipeline.

# %%
fraudulent_ids = baskets[fraud_flags == 1]["ID"]
products[products["basket_ID"].isin(fraudulent_ids)].sort_values("basket_ID")


# %%
# The ``"baskets"`` table itself contains no
# information: we have to bring in features extracted from the corresponding
# products. To do so, we must first vectorize the products lines, so that we
# can aggregate the extracted numeric features, and attach the resulting vector
# to the corresponding ``"basket_ID"`` and thus the corresponding fraud flag.
#
# We do the vectorization with a ``TableVectorizer``. When we run the
# cross-validation, the ``"baskets"`` table (which we marked as X) will be
# split in train and test sets. To fit our ``TableVectorizer``, we want to use
# only products that belong to a basket in the train set, not one in the test
# set. So we do a semi-join to exclude any products that do not belong to a
# basket that can be found in the ``"baskets"`` table currently being processed
# by the pipeline.

# %%

# Note: using deferred or even defining a function is completely optional here,
# because the computation it contains involves no control flow or assignments.
# we only do it to simplify the computation graph by collapsing several
# operations into a single function call.


@skrub.deferred
def filter_products(products, baskets):
    return products[products["basket_ID"].isin(baskets["ID"])]


products = filter_products(products, baskets)

# %%
# We saw before that a transformer can be restricted to a subset of columns in
# a dataframe. The ``cols`` argument can be a column name, a list of column
# names, or a skrub selector. Those selectors are similar to Polars or Ibis
# selectors: they can be used to select columns by name (including glob and
# regex patterns), dtype or other criteria, and they can be combined with the
# same operators as Python sets. Here we want to vectorize all columns in the
# ``"products"`` table, _except_ the ``"basket_ID"`` column which we will need
# for joining.

# %%
from skrub import selectors as s

vectorizer = skrub.TableVectorizer(high_cardinality=skrub.StringEncoder(n_components=8))
vectorized_products = products.skb.apply(vectorizer, cols=s.all() - "basket_ID")


# %%
# Now that we vectorized the product rows, we can aggregate them by basket ID
# before joining on the ``"baskets"`` table.

# %%

# Here also deferred is optional and is used to simplify the display of the
# computation graph.


@skrub.deferred
def join_baskets_products(baskets, vectorized_products):
    aggregated_products = (
        vectorized_products.groupby("basket_ID").agg("mean").reset_index()
    )
    joined = baskets.merge(
        aggregated_products, left_on="ID", right_on="basket_ID"
    ).drop(columns=["ID", "basket_ID"])
    return joined


features = join_baskets_products(baskets, vectorized_products)
features

# %%
# Now we can add the supervised learner, the final step in our pipeline.

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

predictions = features.skb.apply(HistGradientBoostingClassifier(), y=fraud_flags)
predictions

# %%
# We can evaluate our pipeline.

# %%
cv_results = predictions.skb.cross_validate(scoring="roc_auc", verbose=1, n_jobs=4)
cv_results

# %%
cv_results["test_score"]


# %%
# If we are happy with the cross-validation scores, we also probably want to
# fit the pipeline on the data we have, and store it so we can apply it later,
# say to next week's transactions.

# %%
estimator = predictions.skb.get_estimator(fitted=True)
estimator

# %%
# Normally we would save it in the file; here we simulate that by pickling the
# model into a string so that our example notebook does not need to access the
# filesystem.

# %%
import pickle

saved_model = pickle.dumps(estimator)

# %%
# Let us say we got some new data. (This is not truly new data, we just grabbed
# an example basket from the original data, just for illustration.)

# %%
new_baskets = pd.DataFrame({"ID": [21243]})
new_products = pd.DataFrame(
    {
        "basket_ID": [21243, 21243],
        "item": ["COMPUTER PERIPHERALS ACCESSORIES", "FULFILMENT CHARGE"],
        "cash_price": [299, 7],
        "make": ["SAMSUNG", "RETAILER"],
        "model": ["SAMSUNG GALAXY WATCH 3 BLUETOOTH 41MM STAINLESS ST", "RETAILER"],
        "goods_code": ["238905679", "FULFILMENT"],
        "Nbr_of_prod_purchas": [1, 1],
    }
)
new_products

# %%
# We can now load the model and make a prediction

# %%
loaded_model = pickle.loads(saved_model)
loaded_model.predict({"baskets": new_baskets, "products": new_products})

# %%
# The query basket is classified as not fraudulent.

# %%
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
# - ``.skb.concat_horizontal``, ``.skb.drop`` and ``.skb.select``, more skrub selectors
# - ``.skb.freeze_after_fit`` (niche / very advanced)
