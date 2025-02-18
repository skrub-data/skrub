"""
Building complex tabular pipelines
==================================

Skrub provides utilities to build complex, flexible machine-learning pipelines.
They solve several problems that are not easily addressed with the standard
scikit-learn tools such as the ``Pipeline`` and ``ColumnTransformer``.

**Multiple tables:** we need to extract information from several tables
of different shapes (for example, we may have "Customers", "Orders" and
"Products" tables). But scikit-learn estimators (including the ``Pipeline``)
expect their input to be a single design matrix ``X`` and an array of targets
``y`` in which each row corresponds to a sample.

**DataFrame wrangling:** we need operations on dataframes such as aggregations
and joins. Some transformations only apply to some of the columns in a table.
Handling this with scikit-learn's ``FunctionTransformer``, ``Pipeline``,
``ColumnTransformer`` and ``FeatureUnion`` quickly becomes verbose and
difficult to maintain.

**Iterative development:** declaring all the steps in a pipeline before fitting
it to see the result results in a slow development cycle in which errors are
discovered late. We want a more interactive process where we immediately obtain
previews of the intermediate results (or errors).

**Hyperparameter tuning:** many choices such as estimators, hyperparameters,
even the architecture of the pipeline can be informed by validation scores.
Specifying the grid of hyperparameters separately from the model (as is done
for ``GridSearchCV`` & co) is very difficult for complex pipeline, especially
when they involve nested choices (eg choose between 2 different estimators and
tune the hyperparameters of those estimators).

Skrub can help us tackle these challenges. In this example, we show a pipeline
to handle a dataset with 2 tables. Despite being very simple, this pipeline
would be difficult to implement, validate and deploy correctly without skrub.
We show the script with minimal comments to motivate the tools that are
explained in detail in subsequent examples.

"""

# %%
# The credit fraud dataset
# ------------------------
#
# This dataset comes from an e-commerce website. We have a set of "baskets",
# orders that have been placed with the website. Some of those orders were
# fraudulent: the customer made a payment that was later declined by the credit
# card company. Our task is to detect which baskets correspond to a fraudulent
# transaction.
#
# The ``baskets`` table only contains a basket ID and the flag indicating if it
# was fraudulent or not.

# %%
import skrub
import skrub.datasets

dataset = skrub.datasets.fetch_credit_fraud()
skrub.TableReport(dataset.baskets)

# %%
# Each basket contains one or more products. Each row in the ``products`` table
# corresponds to a type of product that was present in a basket. Products can
# be associated with the corresponding basket through the ``"basket_ID"``
# column.

# %%
skrub.TableReport(dataset.products)

# %%
# A data-processing challenge
# ----------------------------
#
# Our end-goal is to fit a supervised learner (a
# ``HistGradientBoostingClassifier``) to predict the fraud flag. To do this, we
# need to build a design matrix in which each row corresponds to a basket (and
# thus to a value in the ``fraud_flag`` column). At the moment, our ``baskets``
# table only contains IDs. We need to enrich it by adding features constructed
# from the actual contents of the baskets, that is, from the ``products``
# table.
#
# As the ``products`` table contains strings and categories (such as
# ``"SAMSUNG"``), we need to vectorize those entries to extract numeric
# features. This is easily done with skrub's ``TableVectorizer``. As each
# basket can contain several products, all the product lines corresponding to a
# basket then need to be aggregated, in order to produce a single feature
# vector that can be attached to the basket (associated with a fraud flag) and
# used to train our ``HistGradientBoostingClassifier``.
#
# Thus the general structure of the pipeline looks like this:
#
# .. image:: ../../_static/credit_fraud_diagram.svg
#    :width: 300
#
# We can see the difficulty: the products need to be aggregated before joining
# to ``baskets``, and in order to compute a meaningful aggregation, they must
# be vectorized *before* the aggregation. So we have a ``TableVectorizer`` to
# fit on a table which does not (yet) have the same number of rows as the
# target ``y`` — something that the scikit-learn ``Pipeline``, with its
# single-input, linear structure, does not accommodate.
#
# We can fit it ourselves, outside of any pipeline with something like::
#
#     vectorizer = skrub.TableVectorizer()
#     vectorized_products = vectorizer.fit_transform(products)
#
# However, because it is dissociated from the main estimator which handles
# ``X`` (the baskets), we have to manage this transformer ourselves. We lose
# the usual scikit-learn machinery for grouping all transformation steps,
# storing fitted estimators, splitting the input data and cross-validation, and
# hyper-parameter tuning.
#
# Moreover, we later need some pandas code to perform the aggregation and join::
#
#     aggregated_products = (
#        vectorized_products.groupby("basket_ID").agg("mean").reset_index()
#     )
#     baskets = baskets.merge(
#         aggregated_products, left_on="ID", right_on="basket_ID"
#     ).drop(columns=["ID", "basket_ID"])
#
#
# Again, as this transformation is not in a scikit-learn estimator, we have to
# keep track of it ourselves so that we can later apply to unseen data, which
# is error-prone, and we cannot tune any choices (like the choice of the
# aggregation function).
#
# To cope with these difficulties, skrub provides an alternative way to build
# more flexible pipelines.

# %%
# A solution with skrub
# ---------------------
#
# Here we just show the solution. Subsequent examples dive into the details.
# TODO TODO TODO
#

# %%
# Declare inputs to the pipeline

# %%
products = skrub.var("products", dataset.products)
baskets = skrub.var("baskets", dataset.baskets[["ID"]]).skb.mark_as_x()
fraud_flags = skrub.var("fraud_flags", dataset.baskets["fraud_flag"]).skb.mark_as_y()

# %%
# Access to the dataframe's usual API; interactive preview of intermediate
# results Note below we are using ``products`` and ``baskets`` as if they were
# a pandas DataFrames

# %%
products = products[products["basket_ID"].isin(baskets["ID"])]
products = products.assign(
    total_price=products["Nbr_of_prod_purchas"] * products["cash_price"]
)
products

# %%
# Easily specify a hyperparameter grid

# %%
n = skrub.choose_int(5, 15, log=True, name="n_components")
encoder = skrub.choose_from(
    {
        "MinHash": skrub.MinHashEncoder(n_components=n),
        "LSA": skrub.StringEncoder(n_components=n),
    },
    name="encoder",
)
vectorizer = skrub.TableVectorizer(high_cardinality=encoder)

# %%
# Easily apply estimators to a subset of columns

# %%
from skrub import selectors as s

vectorized_products = products.skb.apply(vectorizer, cols=s.all() - "basket_ID")

# %%
# Data-wrangling and multiple-table operations as part of the pipeline

# %%
aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()
baskets = baskets.merge(aggregated_products, left_on="ID", right_on="basket_ID").drop(
    columns=["ID", "basket_ID"]
)

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="learning_rate")
)
predictions = baskets.skb.apply(hgb, y=fraud_flags)
predictions

# %%
# TODO link to to full report

# %%
# Perform hyperparameter search or cross-validation

# %%
search = predictions.skb.get_randomized_search(
    scoring="roc_auc", n_iter=16, n_jobs=4, random_state=0, fitted=True
)
search.get_cv_results_table()

# %%
search.plot_parallel_coord()
