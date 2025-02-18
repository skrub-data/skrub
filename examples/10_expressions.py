"""
Building complex tabular pipelines
==================================

Skrub provides utilities to build complex, flexible machine-learning pipelines.
They solve several problems that are not easily addressed with the standard
scikit-learn tools such as the ``Pipeline`` and ``ColumnTransformer``.

**Multiple tables:** a machine-learning estimator may need to transform and
extract information from several tables of different shapes (for example, we
may have "Customers", "Orders" and "Products" tables). But scikit-learn
estimators (including the ``Pipeline``) expect their input to be a single
design matrix ``X`` and an array of targets ``y`` in which each row corresponds
to a sample. They do not easily accommodate operations that change the number
of rows such as aggregating or filtering, or that combine several tables, such
as joins.

**DataFrame wrangling:** the required transformations often involve a mix of
scikit-learn estimators to fit, and of operations on dataframes such as
aggregations and joins. Often, transformations should only be applied to some
of the columns in the input table. These requirements can be met using
scikit-learn's ``FunctionTransformer``, ``Pipeline``, ``ColumnTransformer`` and
``FeatureUnion`` but this can become verbose and difficult to maintain.

**Iterative development:** declaring all the steps in a pipeline before fitting
it to see the result can result in a slow development cycle, in which mistakes
in the early steps of the pipeline are only discovered later, when we fit the
pipeline. We would like a more interactive process where we immediately obtain
previews of the intermediate results (or errors).

**Hyperparameter tuning:** a machine-learning pipeline involves many choices,
such as which tables to use, which features to construct and include, which
estimators to fit, and the estimators' hyperparameters. We want to provide a
range of possible outcomes for these choices, and use validation scores to
select the best option for each (hyperparameter tuning). Scikit-learn offers
``GridSearchCV``, ``RandomizedSearchCV`` and their halving counterparts to
perform the hyperparameter tuning. However, the grid of possible
hyperparameters must be provided separately from the pipeline itself. This can
get cumbersome for complex pipelines, especially when we want to tune not only
simple hyperparameters but also more structural aspects of the pipeline such as
which estimators to use.

Skrub can help us tackle these challenges. In this example, we show a pipeline
to handle a dataset with 2 tables. Despite being very simple, this pipeline
would be difficult to implement, validate and deploy correctly without skrub.
We leave out hyperparameter tuning, which is covered in the
`next example <11_hyperparameters.html>`_.
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
# Each basket contains one or more products. We have a ``products`` table
# detailing the actual content of each basket. Each row in the ``products``
# table corresponds to a type of product that was present in the basket
# (multiple units may have been bought, which is why there is a
# ``"Nbr_of_prod_purchas"`` column). Products can be associated with their
# basket through the ``"basket_ID"`` column.

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
# Skrub expressions
# -----------------
#
# The way we build a machine-learning pipeline in skrub is somewhat different
# from scikit-learn. We do not create the complete pipeline ourselves by
# providing an explicit list of transformation steps. Rather, we manipulate
# objects that represent intermediate results in our computation. They record
# the different operations we perform on them (such as applying operators or
# calling methods), which allows to later retrieve the whole sequence of
# operations as a scikit-learn estimator that can be fitted and applied to
# unseen data.
#
# Because those objects encapsulate a computation, which can be evaluated to
# produce a result, we refer to them as "expressions". The simplest expressions
# are variables, which represent inputs to our pipeline such as the
# ``"products"`` table. Those can be combined with operators and function calls
# to build up more complex expressions.
#
# The first step is therefore to declare the inputs to our pipeline. This can
# be done with ``skrub.var``. For example, we know our system will need to
# handle a ``"products"`` table. So we create a ``"products"`` variable:

# %%
products = skrub.var("products", dataset.products)

# %%
# The variable is given a ``name`` (``"products"``) and a ``value`` (here, the
# dataframe ``dataset.products``). The provided ``value`` is used to compute
# results as we build up the pipeline, and as a source of data for running
# cross-validation and hyperparameter selection (shown later).
#
# (passing a ``value`` is recommended but optional: we can also build our
# pipeline without actually running any computations and execute it later.)
#
# We can then apply transformations to ``products`` to start building up our
# pipeline.
#
# For example, we can write:

# %%
with_total = products.assign(
    total_price=products["Nbr_of_prod_purchas"] * products["cash_price"]
)
with_total

# %%
# Note the added "total_price" column in the result above.
#
# The ``with_total`` object above is a new expression, which encapsulates the
# computation of a new dataframe in which a ``"total_price"`` column has been
# added to the ``"products"`` input.
#
# When we display ``with_total``, we see the result of that computation on the
# data we provided: a pandas DataFrame. But ``with_total`` itself is *not* a
# pandas DataFrame: it is a skrub expression, which stores the computation
# steps needed to produce the result DataFrame. This is essential and enables
# us to later apply those same transformations to some new, unseen data (eg,
# next week's e-commerce transactions, for which we will need predictions as
# well).
#
# We can see a graphical representation of that computation by clicking the
# ``▶ <CallMethod 'assign'>``
# dropdown above, or with the ``.skb.draw_graph()`` method:

# %%
with_total.skb.draw_graph()

# %%
# If we want to access the actual ``DataFrame``, ie the result of evaluating the
# ``with_total`` expression, we can use:

# %%
with_total.skb.eval()

# %%
# We can also pass new bindings for the variables contained in the expression
# to ``eval()``, to get the result for some new inputs.

# %%
with_total.skb.eval({"products": dataset.products.iloc[:3]})

# %%
# (We show it here for
# didactic purposes but in practice you will rarely need to call ``eval``
# yourself)
#
# As we see in the examples above, skrub expressions provide a lot of their
# functionality through the ``.skb`` attribute.
#
# Any other attribute we access will be retrieved from the expression's value,
# so we can use the usual interface of the object we are computing. For example
# here ``with_total`` evaluates to a pandas ``DataFrame`` so we can add more
# computation steps by manipulating it just like we would manipulate a pandas
# ``DataFrame``:

# %%
with_total["item"].str.split(expand=True).rename(columns="item_{}".format)

# %%
# Note that the result of ``with_total['item']`` is not a ``DataFrame`` but a
# ``Series``. More generally, the value of an expression does not have to be a
# ``DataFrame``, it can be anything we want:

# %%
e = (skrub.var("left", "hello") + ", " + skrub.var("right", "world") + "!").title()
e

# %%
e.skb.eval({"right": "skrub"})

# %%
# (``title()`` is a method of Python strings for converting them to title case.)

# %%
# Identifying X and y
# -------------------
#
# Now we have introduced skrub expressions we can get back to our credit fraud
# pipeline.
#
# In addition to the products, our dataset also contains the design matrix
# (which at first, only contains basket IDs) and the fraud flags, so we need to
# create variables for those.
#
# Moreover, we need to mark those variables as the design matrix and the
# prediction targets respectively, so that skrub can know which tables to split
# when running cross-validation or hyperparameter search. We can do this with
# ``.mark_as_x()`` and ``.mark_as_y()``:

# %%
baskets_df = dataset.baskets[["ID"]]  # just a regular dataframe

# mark_as_x() means
# 'this is the feature matrix which needs to be split during cross-validation'
baskets = skrub.var("baskets", baskets_df).skb.mark_as_x()

# Note: a slightly shorter way is to use the shorthand `skrub.X`:
#
# baskets = skrub.X(baskets_df)
#
# is equivalent to:
#
# baskets = skrub.var("X", baskets_df).skb.mark_as_x()

# %%
# similarly for the targets:

# %%
fraud_flags = skrub.var("fraud_flags", dataset.baskets["fraud_flag"]).skb.mark_as_y()

# %%
# Applying scikit-learn estimators
# --------------------------------
#
# Summarizing the above, we now have the 3 variables that represent the 3
# inputs to our pipeline:

# %%
products = skrub.var("products", dataset.products)
baskets = skrub.var("baskets", dataset.baskets[["ID"]]).skb.mark_as_x()
fraud_flags = skrub.var("fraud_flags", dataset.baskets["fraud_flag"]).skb.mark_as_y()

# %%
# Before we start transforming the products table, we do a semi-join on the
# design matrix. This way, when we split the baskets table in cross-validation,
# we will be sure to only use products that correspond to baskets in the
# training set during training.

# %%
products = products[products["basket_ID"].isin(baskets["ID"])]

# %%
# Next, as mentioned before, we need to vectorize the products (extract numeric
# features) before we can aggregate them.
#
# We can use ``.skb.apply()`` to apply a ``TableVectorizer``. This method
# accepts:
#
# - the scikit-learn estimator to apply
# - optionally, a ``cols`` argument, which specifies that the transformation
#   should only be applied to some of the columns in a dataframe. This can be a
#   column name, a list of column names, or more complex selectors that we can
#   build with ``skrub.selectors``:

# %%
from skrub import selectors as s

product_vectorizer = skrub.TableVectorizer(
    high_cardinality=skrub.StringEncoder(n_components=5)
)

# We want to vectorize all columns except "basket_ID", which we will need to
# join with the baskets table.
vectorized_products = products.skb.apply(product_vectorizer, cols=s.all() - "basket_ID")
vectorized_products

# %%
# Now that we vectorized the products, we can aggregate them and join to the
# baskets table

# %%
aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()
baskets = baskets.merge(aggregated_products, left_on="ID", right_on="basket_ID").drop(
    columns=["ID", "basket_ID"]
)

# %%
# Finally, we have features and targets — we can fit our gradient boosting
# classifier. Note that when we are using a supervised estimator, we need to
# also pass ``y`` to ``.skb.apply``. In this case it is the fraud flag.

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

predictions = baskets.skb.apply(HistGradientBoostingClassifier(), y=fraud_flags)
predictions

# %%
# Getting a full report for the pipeline
# --------------------------------------
#
# We are now done building up our pipeline: we have reached the last step,
# which evaluates to the predicted fraud flags.
#
# We can ask it for a full report. It is an HTML output in which we can
# inspect each intermediate result in the computation. It does not display
# inline in a notebook but rather creates a folder on the filesystem, so we do
# not run it in this example but you can see the result
# `here <../_static/credit_fraud_report/index.html>`_.
# Click on the node in the graph to see details. The command to obtain it is::
#
#     predictions.skb.full_report()
#
# `See the output <../_static/credit_fraud_report/index.html>`_.

# %%
# Obtaining and fitting an estimator
# ----------------------------------
#
# We can now ask skrub for different things such as cross-validation scores or an
# estimator that we can fit, serialize, and later apply to unseen data (shown below).
#
# To cross-validate our pipeline on the data we already provided:

# %%
predictions.skb.cross_validate(scoring="roc_auc", n_jobs=4)

# %%
# To obtain an estimator with a similar API to scikit-learn estimator, that we
# can fit and store:

# %%
# If we pass fitted=True, the estimator is fitted on the data we provided
estimator = predictions.skb.get_estimator(fitted=True)

# %%
# We can now store it on disk (here we use an in-memory string for
# demonstration purposes):

# %%
import pickle

stored = pickle.dumps(estimator)


# %%
# And load it to get predictions on some new data:

# %%
loaded = pickle.loads(stored)

# simulate getting new data points for which we need a prediction
new_baskets, new_products = dataset.baskets[["ID"]], dataset.products

# Get a prediction
loaded.predict({"baskets": new_baskets, "products": new_products})

# %%
# Note the difference with the scikit-learn API: usually we write
#
# ``estimator.predict(X=df)``
#
# But as the estimators constructed by skrub may accept multiple tables, here
# we pass a dictionary of values. Each key corresponds to the name of a
# variable we used in our expression:
#
# ``estimator.predict({"name": value, ...})``
#
# Similarly, if we wanted to fit the estimator to some new data, we would
# write::
#
#     estimator.fit(
#         {
#             "products": training_products,
#             "baskets": training_baskets,
#             "fraud_flags": training_fraud_flags,
#         }
#     )

# %%
# Conclusion
# ----------
#
# This has been a brief overview of some of the features of skrub expressions.
# A very important aspect which we completely left out is model selection /
# hyperparameter tuning, which we cover in the next example.
#
# This example is long and introduces many concepts, but if we regroup all the
# code needed to build a full predictive pipeline for our credit fraud problem
# we see that it is actually quite short::
#
#     from sklearn.ensemble import HistGradientBoostingClassifier
#
#     import skrub
#     import skrub.datasets
#     from skrub import selectors as s
#
#     dataset = skrub.datasets.fetch_credit_fraud()
#
#     products = skrub.var("products", dataset.products)
#     baskets = skrub.var("baskets", dataset.baskets[["ID"]]).skb.mark_as_x()
#     fraud_flags = skrub.var(
#         "fraud_flags", dataset.baskets["fraud_flag"]
#     ).skb.mark_as_y()
#
#     products = products[products["basket_ID"].isin(baskets["ID"])]
#
#     product_vectorizer = skrub.TableVectorizer(
#         high_cardinality=skrub.StringEncoder(n_components=5)
#     )
#
#     vectorized_products = products.skb.apply(
#         product_vectorizer, cols=s.all() - "basket_ID"
#     )
#
#     aggregated_products = (
#         vectorized_products.groupby("basket_ID").agg("mean").reset_index()
#     )
#     baskets = baskets.merge(
#         aggregated_products, left_on="ID", right_on="basket_ID"
#     ).drop(columns=["ID", "basket_ID"])
#
#     predictions = baskets.skb.apply(HistGradientBoostingClassifier(), y=fraud_flags)
#
#     predictions.skb.cross_validate(scoring="roc_auc", n_jobs=4)

# %%
