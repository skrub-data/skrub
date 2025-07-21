"""

Multiples tables: building machine learning pipelines with DataOps
==================================================================

In this example, we show how to build a DataOps plan to handle
pre-processing, validation and hyperparameter tuning of a dataset with **multiple
tables**.

We consider the credit fraud dataset, which contains two tables: one for
baskets (orders) and one for products. The goal is to predict whether a basket
is fraudulent or not, based on the products it contains.

.. |choose_from| replace:: :func:`skrub.choose_from`
.. |choose_int| replace:: :func:`skrub.choose_int`
.. |choose_float| replace:: :func:`skrub.choose_float`
.. |MinHashEncoder| replace:: :class:`skrub.MinHashEncoder`
.. |StringEncoder| replace:: :class:`skrub.StringEncoder`
.. |TableVectorizer| replace:: :class:`skrub.TableVectorizer`
.. |var| replace:: :func:`skrub.var`
.. |TableReport| replace:: :class:`skrub.TableReport`
.. |HistGradientBoostingClassifier| replace::
   :class:`sklearn.ensemble.HistGradientBoostingClassifier`
.. |make_randomized_search| replace:: :meth:`skrub.Expr.make_randomized_search`

.. currentmodule:: skrub

"""

# %%
# The credit fraud dataset
# ------------------------
#
# This dataset comes from an e-commerce website. We have a set of "baskets"
# (orders that have been placed with the website). The task is to detect which
# orders were fraudulent (the customer never made the payment).
#
# The baskets table contains a basket ID and a flag indicating if the order
# was fraudulent or not.

# %%
import skrub
import skrub.datasets

dataset = skrub.datasets.fetch_credit_fraud()
skrub.TableReport(dataset.baskets)

# %%
# The ``products`` table contains information about the products that have been
# purchased, and the basket they belong to. A basket contains at least one product.
# Products can be associated with the corresponding basket through the "basket_ID"
# column.

# %%
skrub.TableReport(dataset.products)

# %%
# A data-processing challenge
# ----------------------------
# The general structure of the DataOps plan we want to build looks like this:
#
# .. image:: ../../_static/credit_fraud_diagram.svg
#    :width: 300
#
# We want to fit a |HistGradientBoostingClassifier| to predict the fraud
# flag (y). However, since the features for each basket are stored in
# the products table, we need to extract these features, aggregate them
# at the basket level, and merge the result with the basket data.
#
# We can use the |TableVectorizer| to vectorize the products, but we
# then need to aggregate the resulting vectors to obtain a single row per basket.
# Using a scikit-learn Pipeline is tricky because the |TableVectorizer| would be
# fitted on a table with a different number of rows than the target y (the baskets
# table), which scikit-learn does not allow.
#
# While we could fit the |TableVectorizer| manually, this would forfeit
# scikit-learn’s tooling for managing transformations, storing fitted estimators,
# splitting data, cross-validation, and hyper-parameter tuning.
# We would also have to handle the aggregation and join ourselves, likely with
# error-prone Pandas code.
#
# Fortunately, skrub DataOps provide a powerful alternative for building flexible
# plans that address these problems.

# %%
# Building a multi-table DataOps plan
# ------------------------------------
#
# We start by creating skrub variables, which are the inputs to our plan.
# In our example, we create three skrub |var| objects: ``products``, ``baskets``,
# and ``fraud_flags``:

# %%
products = skrub.var("products", dataset.products)
full_baskets = skrub.var("baskets", dataset.baskets)

baskets = full_baskets[["ID"]].skb.mark_as_X()
fraud_flags = full_baskets["fraud_flag"].skb.mark_as_y()

# %%
# We mark the "baskets" variable as ``X`` and the "fraud flags" variable as ``y``
# so that DataOps can use their indices for train-test splitting and cross-validation.
# We then build the plan by applying transformations to those inputs.
#
# Since our DataOps expect dataframes for products, baskets and fraud
# flags, we manipulate those objects as we would manipulate pandas dataframes.
# For instance, we filter products to keep only those that match one of the
# baskets in the ``baskets`` table, and then add a column containing the total
# amount for each kind of product in a basket:

# %%
kept_products = products[products["basket_ID"].isin(baskets["ID"])]
products_with_total = kept_products.assign(
    total_price=kept_products["Nbr_of_prod_purchas"] * kept_products["cash_price"]
)
products_with_total

# %%
# We then build a skrub ``TableVectorizer`` with different choices of
# the type of encoder for high-cardinality categorical or string columns, and
# the number of components it uses.
#
# With skrub, we do not need to specify a grid of hyperparameters separately
# from the pipeline. Instead, within a DataOps plan we can replace a parameter's
# value with one of skrub's ``choose_*``` functions, which indicate the range of
# values we consider during hyperparameter selection. Here, we use |choose_int|
# to choose the number of components for the encoder, and |choose_from| to choose
# the type of encoder to use.

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
# We can restrict the vectorizer to a subset of columns: in our case, we want to
# vectorize all columns except the ``"basket_ID"`` column, which is not a
# feature but a link to the basket it belongs to.

# %%
vectorized_products = products_with_total.skb.apply(
    vectorizer, exclude_cols="basket_ID"
)

# %%
# We then aggregate the vectorized products by basket ID, and then merge the result
# with the baskets table.

# %%
aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()
augmented_baskets = baskets.merge(
    aggregated_products, left_on="ID", right_on="basket_ID"
).drop(columns=["ID", "basket_ID"])

# %%
# Finally, we add a supervised estimator, and use |choose_float| to
# add the learning rate as a hyperparameter to tune.

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="learning_rate")
)
predictions = augmented_baskets.skb.apply(hgb, y=fraud_flags)
predictions

# %%
# And our DataOps plan is complete!
#
# We can now use |make_randomized_search| to perform hyperparameter
# tuning and find the best hyperparameters for our model.

# %%
print(predictions.skb.describe_param_grid())

# %%
search = predictions.skb.make_randomized_search(
    scoring="roc_auc", n_iter=8, n_jobs=4, random_state=0, fitted=True
)
search.results_

# %%
# We can also display the results of the search in a parallel coordinates plot:
search.plot_results()

# %%
# It seems here that using the LSA as an encoder brings better test scores,
# but at the expense of training and scoring time.

# %%
# Conclusion
# ----------
# In this example, we have shown how to build a multi-table machine learning
# pipeline with the skrub DataOps. We have seen how DataOps allow us to use Pandas
# to manipulate dataframes, and how we can build a DataOps plan that can make use
# of multiple tables, and perform hyperparameter tuning on the resulting pipeline.
#
# If you are curious to know more on how to tune hyperparameters using the skrub
# DataOps, please see
# :ref:`Tuning Pipelines example <example_tuning_pipelines>` for an
# in-depth tutorial.
