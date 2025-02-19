"""
Skrub expressions
=================

TODO
"""

# %%
import skrub
import skrub.datasets

dataset = skrub.datasets.fetch_credit_fraud()

# %%
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
