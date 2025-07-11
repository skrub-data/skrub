"""
.. _example_expressions_intro:


Building a predictive model by combining multiple tables with the skrub Data Plan
====================================================================
This example summarizes the use of skrub DataOps to build a ``Learner``:  recording
all the Data Operations involved in building a predictor starting from one or
more tables, and then replays them on unseen data.


"""

# %%
# The credit fraud dataset
# ------------------------
#
# This dataset originates from an e-commerce website and is structured into two tables:
#
# - The "baskets" table contains order IDs, each representing a list of purchased products. 
#   For a subset of these orders (the training set), a flag indicates whether the order was fraudulent. 
#   This fraud flag is the target variable we aim to predict during inference.
# - The "products" table provides the detailed contents of all baskets, including those without a known fraud label.
# The ``baskets`` table contains a basket ID and a flag indicating if the order
# was fraudulent or not.
# We start by loading the ``baskets`` table, and exploring it with the ``TableReport``.
#
# %%
import skrub
import skrub.datasets

dataset = skrub.datasets.fetch_credit_fraud()  # load labeled data
skrub.TableReport(dataset.baskets)

# %%
# We then load the ``products`` table, which contains one row per purchased product.

# %%
skrub.TableReport(dataset.products)

# %%
# Each basket contains at least one product, and products can
# be associated with the corresponding basket through the ``"basket_ID"``
# column.
#
# A design problem: how to combine tables while avoiding leakage?
# ----------------------------
#
# We want to fit a ``HistGradientBoostingClassifier`` to predict the fraud
# flag (or ``y``). To do so, we build a design matrix where each row corresponds
# to a basket, and we want to add features from the ``products`` table to
# each basket.
#
# The general structure of the pipeline looks like this:
#
# .. image:: ../../_static/credit_fraud_diagram.svg
#    :width: 300
#
#
# First, as the ``products`` table contains strings and categories (such as
# ``"SAMSUNG"``), we vectorize those entries to extract numeric
# features. This is easily done with skrub's ``TableVectorizer``. Then, since each
# basket can contain several products, we want to aggregate all the lines in
# ``products`` that correspond to a single basket into a single vector that can
# then be attachet to the basket.
#
# The difficulty is that the vectorized ``products`` should be aggregated before joining
# to ``baskets``, and, in order to compute a meaningful aggregation, must
# be vectorized *before* the aggregation. Thus, we have a ``TableVectorizer`` to
# fit on a table which does not (yet) have the same number of rows as the
# target ``y`` — something that the scikit-learn ``Pipeline``, with its
# single-input, linear structure, does not allow.
#
# We can fit it ourselves, outside of any pipeline with something like::
#
#     vectorizer = skrub.TableVectorizer()
#     vectorized_products = vectorizer.fit_transform(dataset.products)
#
# However, because it is dissociated from the main estimator which handles
# ``X`` (the baskets), we have to manage this transformer ourselves. We lose
# the scikit-learn machinery for grouping all transformation steps,
# storing fitted estimators, cross-validating the data, and tuning hyper-parameters.
#
# Moreover, we might need some Pandas code to perform the aggregation and join.
# Again, as this transformation is not in a scikit-learn estimator, it is error-prone.
# The difficulty is that we have to keep track of it ourselves to apply it later
# to unseen data, and we cannot tune any choices (like the choice of the
# aggregation function).
#
# Fortunately, skrub provides an alternative way to build
# more flexible pipelines.

# %%
# Data Ops make Data Plans
# ------------------------
# In the skrub Data Plan, we do not have an explicit, sequential list of
# transformation steps. Instead, we perform "Data Operations" (or "Data Ops"),
# that is, operations that act on variables and wrap user operations to keep track
# of their parameters.
#
# User operations could be dataframe operations (selection, merge, group by, etc.),
# scikit-learn estimators (such as a RandomForest with its hyperparameters),
# or arbitrary code (for loading data, converting values, etc.).
#
# As we perform operations on skrub variables, the Data Plan records each DataOp
# and its parameters. This record can later be synthesized into a standalone object
# called a "learner", which can replay these operations on unseen data, ensuring
# that the same operations and parameters are used.
#
# In a skrub Data Plan, we manipulate skrub objects representing
# intermediate results. The plan is built implicitly as we perform
# operations (such as applying operators or calling functions) on those
# objects.

# %%
# We start by creating skrub variables, which are the inputs to our Data Plan.
# In our example, we create three variables: "products", "baskets", and "fraud flags":

# %%
products = skrub.var("products", dataset.products)
full_baskets = skrub.var("baskets", dataset.baskets)

baskets = full_baskets[["ID"]].skb.mark_as_X()
fraud_flags = full_baskets["fraud_flag"].skb.mark_as_y()

# %%
# They are given a name and an (optional) initial
# value, used to show previews of the result of each Data Operation, detect errors
# early, and provide data for cross-validation and hyperparameter search.
# In a later example, we will show how to use subsampling to speed up the
# generation of previes.
#
# Then, the Data Plan is built by applying Data Operations to those inputs, that
# is, by performing user operations that have been wrapped in a Data Op.
#
# Above, ``mark_as_X()`` and ``mark_as_y()`` indicate that the baskets and
# flags are respectively our design matrix and target variables, that
# should be split into training and testing sets for cross-validation. Here,
# they are direct inputs to the Data Plan, but any
# intermediate result could be marked as X or y.
#
# By setting products, baskets and fraud_flags as skrub variables, we can manipulate
# those objects as if they were dataframes, while keeping track of all the operations
# that are performed on them.
#
# For instance, we can filter products to keep only those that match one of the
# baskets in the ``baskets`` table, and then add a column containing the total
# amount for each kind of product in a basket:

# %%
kept_products = products[products["basket_ID"].isin(baskets["ID"])]
products_with_total = kept_products.assign(
    total_price=kept_products["Nbr_of_prod_purchas"] * kept_products["cash_price"]
)
products_with_total

# %%
# We see previews of the output of intermediate results. For
# example, the added ``"total_price"`` column is in the output above.
# The "Show graph" dropdown at the top allows us to check the
# structure of the Data Plan and all the Data Ops it contains.
#
# .. note::
#
#    We recommend to assign each new skrub Data Operation to a new variable name,
#    as is done above. For example ``kept_products = products[...]`` instead of
#    reusing the name ``products = products[...]``. This makes it easy to
#    backtrack to any step of the Data Plan and change the subsequent steps, and
#    can avoid ending up in a confusing state in jupyter notebooks when the
#    same cell might be re-executed several times.
#
# With skrub, we do not need to specify a grid of hyperparameters separately
# from the Data Plan. Instead, we replace a parameter's value with a skrub
# "choice" which indicates the range of values we consider during
# hyperparameter selection.
#
# Skrub choices can be nested arbitrarily. They are not restricted to
# parameters of a scikit-learn estimator, but can be anything: choosing
# between different estimators, arguments to function calls, whole sections of
# the Data Plan etc.
#
# In-depth information about choices and hyperparameter/model selection is
# provided in the :ref:`Tuning Data Plans example <example_tuning_pipelines>`.
#
# We build a skrub ``TableVectorizer`` with different choices of:
# the type of encoder for high-cardinality categorical or string columns, and
# the number of components it uses.

# %%
n = skrub.choose_int(5, 15, name="n_components")
encoder = skrub.choose_from(
    {
        "MinHash": skrub.MinHashEncoder(n_components=n),
        "LSA": skrub.StringEncoder(n_components=n),
    },
    name="encoder",
)
vectorizer = skrub.TableVectorizer(high_cardinality=encoder)

# %%
# A transformer does not have to apply to the full dataframe; we can
# restrict it to some columns, using the ``cols`` or ``exclude_cols``
# parameters. In our example, we vectorize all columns except the ``"basket_ID"``.

# %%
vectorized_products = products_with_total.skb.apply(
    vectorizer, exclude_cols="basket_ID"
)

# %%
# Having access to the underlying dataframe's API, we can perform the
# data-wrangling we need. Those transformations are being implicitly recorded
# as Data Ops in our Data Plan.

# %%
aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()
augmented_baskets = baskets.merge(
    aggregated_products, left_on="ID", right_on="basket_ID"
).drop(columns=["ID", "basket_ID"])

# %%
# We can actually ask for a full report of the Data Plan and inspect the
# results of each DataOp::
#
#     predictions.skb.full_report()
#
# This produces a folder on disk rather than displaying inline in a notebook so
# we do not run it here. But you can
# `see the output here <../../_static/credit_fraud_report/index.html>`_.
#
# Finally, we add a supervised estimator:

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="learning_rate")
)
predictions = augmented_baskets.skb.apply(hgb, y=fraud_flags)
predictions

# %%
# And our Data Plan is complete!
#
# From the choices we inserted at different locations in our Data Plan, skrub
# can build a grid of hyperparameters and run the hyperparameter search for us,
# backed by scikit-learn's ``GridSearchCV`` or ``RandomizedSearchCV``.

# %%
print(predictions.skb.describe_param_grid())

# %%
search = predictions.skb.get_randomized_search(
    scoring="roc_auc", n_iter=8, n_jobs=4, random_state=0, fitted=True
)
search.results_

# %%
# We can also run a cross validation, using the first choices defined in the ``choose``
# objects:

predictions.skb.cross_validate(scoring="roc_auc", verbose=1, n_jobs=4)

# %%
# We can also display a parallel coordinates plot of the results.
#
# In a parallel coordinates plot, each line corresponds to a combination
# of hyperparameter (choices) values, followed by the corresponding test
# scores, and training and scoring computation durations.
# Different columns show the hyperparameter values.
#
# By **clicking and dragging the mouse** on any column, we can restrict the
# set of lines we see. This allows quickly inspecting which hyperparameters are
# important, which values perform best, and potential trade-offs between the quality
# of predictions and computation time.

# %%
search.plot_results()

# %%
# It seems here that using the LSA as an encoder brings better test scores,
# but at the expense of training and scoring time.
#
# From the Data Plan to the learner
# -----------
# The learner is a standalone object that can replay all the Data Ops recorded in
# the Data Plan, and can be used to make predictions on new, unseen data. The
# learner can be saved and loaded, allowing us to use it later without having to
# rebuild the Data Plan.
# We would usually save the learner in a binary file, but to avoid accessing the
# filesystem with this example notebook, we serialize the learner in memory instead.
import pickle

saved_model = pickle.dumps(search.best_pipeline_)  # This gives us our learner

# %%
# Let's say we got some new data, and we want to use the learner we just saved
# to make predictions on them:
new_data = skrub.datasets.fetch_credit_fraud(split="test")
new_baskets = new_data.baskets[["ID"]]
new_products = new_data.products

# %%
# Our learner expects the same variable names as the training Data Plan, which is why
# we pass a dictionary that contains new dataframes and the same variable:
loaded_model = pickle.loads(saved_model)
loaded_model.predict({"baskets": new_baskets, "products": new_products})

# %%
# Conclusion
# ----------
#
# If you are curious to know more on how to build your own complex, multi-table
# Data Plans with easy hyperparameter tuning and transforming them into reusable
# learners, please see the next examples for an in-depth tutorial.
