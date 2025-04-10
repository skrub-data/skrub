"""
Supervised estimators and cross-validation
------------------------------------------

We can use ``.skb.apply()`` to add scikit-learn transformers to the computation,
but also to add a supervised learner like ``HistGradientBoostingClassifier``
or ``LogisticRegression``.

When we have built a full estimator, we want to run cross-validation to
measure its performance. We may also want to tune hyperparameters (shown in
the next example).

Skrub can run the cross-validation for us. In order to do so, it needs to
split the dataset into training and testing sets. For this to happen, we need
to tell skrub which items in our pipeline constitute the feature matrix ``X``
and the targets (labels, outputs) ``y``.

Indeed, an estimator built from a skrub expression can accept inputs that are
not yet neatly organized into correctly-aligned ``X`` and ``y`` matrices.
There may be some steps (such as loading data, performing joins or
aggregations, separating features and targets from a single source) that need
to be performed in order to construct the design matrix the targets.

To indicate which intermediate results to split, we call ``.skb.mark_as_X()``
and ``.skb.mark_as_y()`` on the appropriate objects:
"""

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

import skrub.datasets

dataset = skrub.datasets.fetch_employee_salaries()

full_data = skrub.var("data", dataset.employee_salaries)

employees = full_data.drop(columns="current_annual_salary").skb.mark_as_X()
salaries = full_data["current_annual_salary"].skb.mark_as_y()

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
# If you unfold the dropdown to see the computation graph, you will see the
# inner nodes that have been marked as ``X`` and ``y``: these nodes are highlighted
# in blue and red, respectively. Furthermore, the ``data`` is an input variable,
# so it is marked with a double box.
#
# To perform
# cross-validation, skrub first runs all the prior steps until it has computed
# ``X`` and ``y``. Then, it splits those (with the usual scikit-learn
# cross-validation tools, as is done in ``sklearn.cross_validate``) and runs
# the rest of the computation inside of the cross-validation loop.
#

# %%
predictions.skb.cross_validate()


# %%
# In the simplest case where we have ``X`` and ``y`` available right from the
# start, we can indicate that simply by creating the input variables with
# ``skrub.X()`` and ``skrub.y()`` instead of ``skrub.var()``: ``skrub.X()`` is
# syntactic sugar for ``skrub.var("X").skb.mark_as_X()``.

# %%
# The construction of ``X`` and ``y`` must be done before splitting the samples
# into cross-validation folds; it happens outside of the cross-validation loop.
#
# This means that any step we perform in this part of the computation has
# access to the full data (training and test sets). We should not use
# estimators that learn from the data before reaching the cross-validation
# loop, or we might obtain optimistic scores due to data leakage.
# Join operations that involve aggregations are not safe either, because they
# may involve information from the test set. Finally, no
# choices or hyperparameters can be tuned in this part of the computation
# (tuning is discussed in more detail in the next example).
#
# Therefore, we should build ``X`` and ``y`` at the very start of the
# computation and use ``mark_as_X()`` and ``mark_as_y()`` as soon as possible
# -- as soon as we have separate ``X`` and ``y`` tables that are aligned and
# have one row per sample. In particular, we should use ``mark_as_X()`` before
# doing any feature extraction and selection.
#
# Skrub will let us apply transformers before reaching ``mark_as_X`` because
# there are special cases where we know it is safe and faster to do so, but in
# general we should be careful and remember to separate features and targets
# and then use ``mark_as_X()`` as soon as possible.


# %%
# A full example
# --------------
#
# Now we know enough about skrub expressions to create a full estimator for a
# realistic example and evaluate its performance.
#
# We finally come back to the credit fraud dataset from the previous example.

# %%
import skrub.datasets

dataset = skrub.datasets.fetch_credit_fraud()

# %%
# The ``baskets`` are orders on a e-commerce website. Note that in the drop-down
# menu, ``"baskets"`` is both an input variable and a table that is
# marked as ``X``, and therefore it is colored blue and has a double box.

# %%
baskets = skrub.var("baskets", dataset.baskets[["ID"]]).skb.mark_as_X()
baskets

# %%
# Each basket is associated with a fraud flag. The ``"fraud_flag"`` column in the
# ``baskets`` table is red as it is marked as ``y``.

# %%
fraud_flags = skrub.var("fraud_flags", dataset.baskets["fraud_flag"]).skb.mark_as_y()
fraud_flags

# %%
# Each basket contains one or several products. The ``"basket_ID"`` column
# in the ``"products"`` refers to the ``"ID"`` column in the ``"baskets"``
# table. Given the information we have about a basket, we want to predict if
# its purchase was fraudulent.

# %%
products = skrub.var("products", dataset.products)
products

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
# basket that can be found in the ``"baskets"`` table currently being
# processed.

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
# names, or a skrub selector. Skrub selectors are similar to Polars or Ibis
# selectors: they can be used to select columns by name (including glob and
# regex patterns), dtype or other criteria, and they can be combined with the
# same operators as Python sets. Here we have a very simple selection to make:
# we want to vectorize all columns in the ``"products"`` table, _except_ the
# ``"basket_ID"`` column, which we will need for joining. So we can just use the
# ``exclude_cols`` parameter.

# %%
vectorizer = skrub.TableVectorizer(high_cardinality=skrub.StringEncoder(n_components=8))
vectorized_products = products.skb.apply(vectorizer, exclude_cols="basket_ID")


# %%
# Now that we vectorized the product rows, we can aggregate them by basket ID
# before joining on the ``"baskets"`` table.

# %%

# Here, deferred is still optional and is used to simplify the display of the
# computation graph. By commenting it out, you will be able to see all the steps
# that are performed in ``join_baskets_products`` to reach the final result.


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
# We can now add the supervised learner, the final step in our computation graph.

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

predictions = features.skb.apply(HistGradientBoostingClassifier(), y=fraud_flags)
predictions

# %%
# Finally, we can evaluate our estimator.

# %%
cv_results = predictions.skb.cross_validate(scoring="roc_auc", verbose=1, n_jobs=4)
cv_results

# %%
cv_results["test_score"]

# %%
# If we are happy with the cross-validation scores, we might want to
# fit the estimator on the data we have, and save the estimator object
# so that we can use it later, say for next week's transactions.

# %%
estimator = predictions.skb.get_estimator(fitted=True)
estimator

# %%
# Normally we would save it in a file; here we simulate that by pickling the
# model into a string so that our example notebook does not need to access the
# filesystem.

# %%
import pickle

saved_model = pickle.dumps(estimator)

# %%
# Let us say we got some new data, and we want to use the model we just saved
# to make predictions on it.
#

# %%
new_data = skrub.datasets.fetch_credit_fraud(split="test")
new_baskets = new_data.baskets[["ID"]]
new_products = new_data.products
new_products

# %%
# We can then load the saved model to make a prediction on the new data. To do so,
# we need to pass the new data so that the same transformations can be applied to
# it.
# Note that the loaded estimator will expect the same input variables as the
# original pipeline, with the same names: this is why we pass a dictionary that
# contains the new dataframes and the same variable names (``baskets`` and
# ``products``) that we used when we built the original estimator.

# %%
loaded_model = pickle.loads(saved_model)
loaded_model.predict({"baskets": new_baskets, "products": new_products})

# %%
# Conclusion
# ----------
# In this example we have seen how to build a full estimator with skrub expressions,
# including a supervised learner, and how to evaluate it with cross-validation.
# The example included complex data preparation and aggregation steps that involved
# joining dataframes: thanks to the skrub expressions, we could build a full
# pipeline that ensures that no leakage occurs, and we were able to obtain a
# new estimator that we could save and use later.
#
# This is not all there is to the skrub expressions: in the next example we will
# go over hyperparameter tuning, which expressions simplify a lot.
#
# A few more advanced features have not been shown and remain for more
# specialized examples, for example:
#
# - naming nodes, passing the value for any node with the inputs
# - ``.skb.applied_estimator``
# - ``.skb.concat_horizontal``, ``.skb.drop`` and ``.skb.select``, more skrub selectors
# - ``.skb.freeze_after_fit`` (niche / very advanced)
