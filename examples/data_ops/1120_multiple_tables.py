"""
Multiples tables: building machine learning pipelines with DataOps
==================================================================

In this example, we show how to build a DataOps plan to handle
pre-processing, validation and hyperparameter tuning of a dataset with **multiple
tables**.

We consider the credit fraud dataset, which contains two tables: one for
baskets (orders) and one for products. The goal is to predict whether a basket
(a single order that has been placed with the website) is fraudulent or not,
based on the products it contains.

.. currentmodule:: skrub

.. |choose_from| replace:: :func:`skrub.choose_from`
.. |choose_int| replace:: :func:`skrub.choose_int`
.. |choose_float| replace:: :func:`skrub.choose_float`
.. |MinHashEncoder| replace:: :class:`~skrub.MinHashEncoder`
.. |StringEncoder| replace:: :class:`~skrub.StringEncoder`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |var| replace:: :func:`skrub.var`
.. |TableReport| replace:: :class:`~skrub.TableReport`
.. |HistGradientBoostingClassifier| replace::
   :class:`~sklearn.ensemble.HistGradientBoostingClassifier`
.. |make_randomized_search| replace:: :func:`~skrub.DataOp.skb.make_randomized_search`


"""

# %%
# The credit fraud dataset
# ------------------------
#
# The ``baskets`` table contains a basket ID and a flag indicating if the order
# was fraudulent or not (the customer never made the payment).

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
# In our example, we create two skrub |var| objects: ``products`` and ``baskets``:

# %%
products = skrub.var("products", dataset.products)
baskets = skrub.var("baskets", dataset.baskets)

basket_ids = baskets[["ID"]].skb.mark_as_X()
fraud_flags = baskets["fraud_flag"].skb.mark_as_y()

# %%
# We mark the ``basket_ids`` variable as ``X`` and the ``fraud_flags`` variable as ``y``
# so that DataOps can use their indices for train-test splitting and cross-validation.
# We then build the plan by applying transformations to those inputs.
#
# Since our DataOps expect dataframes for products, baskets and fraud
# flags, we manipulate those objects as we would manipulate pandas dataframes.
# For instance, we filter products to keep only those that match one of the
# baskets in the ``baskets`` table, and then add a column containing the total
# amount for each kind of product in a basket:

# %%
kept_products = products[products["basket_ID"].isin(basket_ids["ID"])]
products_with_total = kept_products.assign(
    total_price=kept_products["Nbr_of_prod_purchas"] * kept_products["cash_price"]
)
products_with_total

# %%
# We then build a skrub ``TableVectorizer`` with different choices of
# the type of encoder for high-cardinality categorical or string columns, and
# the number of components it uses.
#
# With skrub, there’s no need to specify a separate grid of hyperparameters outside
# the pipeline.
# Instead, within a DataOps plan, we can directly replace a parameter’s value using
# one of skrub’s ``choose_*`` functions, which define the range of values to consider
# during hyperparameter selection. In this example, we use |choose_int| to select
# the number of components for the encoder and |choose_from| to select the type
# of encoder.

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
augmented_baskets = basket_ids.merge(
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
# tuning and find the best hyperparameters for our model. Below, we display the
# hyperparameter combinations that define our search space.

# %%
print(predictions.skb.describe_param_grid())

# %%
# |make_randomized_search| returns a :class:`~skrub.ParamSearch` object, which contains
# our search result and some plotting logic.
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
#
# We can get the best performing :class:`~skrub.SkrubLearner` via
# ``best_learner_``, and use it for inference on new data with:

import pandas as pd

new_baskets = pd.DataFrame([dict(ID="abc")])
new_products = pd.DataFrame(
    [
        dict(
            basket_ID="abc",
            item="COMPUTER",
            cash_price=200,
            make="APPLE",
            model="XXX-X",
            goods_code="239246782",
            Nbr_of_prod_purchas=1,
        )
    ]
)
search.best_learner_.predict_proba({"baskets": new_baskets, "products": new_products})
# %%
# Conclusion
# ----------
#
# In this example, we have shown how to build a multi-table machine learning
# pipeline with skrub DataOps. We have seen how DataOps allow us to use familiar
# Pandas operations to manipulate dataframes, and how we can build a DataOps plan
# that works with multiple tables and performs hyperparameter tuning on the
# resulting pipeline.
#
# If you want to learn more about tuning hyperparameters using skrub DataOps, see
# the :ref:`Tuning Pipelines example <example_tuning_pipelines>` for an
# in-depth tutorial.
