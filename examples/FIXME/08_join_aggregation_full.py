"""
AggJoiner on a credit fraud dataset
===================================

In this example, we are tackling a fraudulent loan detection use case.
Because fraud is rare, this dataset is extremely imbalanced, with a prevalence of around
1.4%.

Instead of focusing on arbitrary metrics like accuracy, we will derive a cost function
based on (questionable) assumptions about the data. In a real-world scenario, we would
need to consult with a domain expert within the company to develop a realistic utility
function.

The data consists of two distinct concepts: a "basket," which can be tagged as fraud (1)
or not (0), and a list of "products." Each product has several attributes:

- a category (marked by the column ``"item"``),
- a model (``"model"``),
- a brand (``"make"``),
- a merchant code (``"goods_code"``),
- a price per unit (``"cash_price"``),
- a quantity selected in the basket (``"Nbr_of_prod_purchas"``)

Since the number of products in each basket varies, the creators of this dataset have
chosen to join all products and their attributes with their respective basket. They have
arbitrarily decided to cut off the basket at the 24th product. However, since most
baskets contain only one or two products, a large proportion of the columns are empty.
Therefore, the dataset is very sparse, which is challenging from a machine learning
perspective and also inefficient in terms of memory usage.

.. |AggJoiner| replace::
     :class:`~skrub.AggJoiner`

.. |Joiner| replace::
     :class:`~skrub.Joiner`

.. |TableVectorizer| replace::
     :class:`~skrub.TableVectorizer`

.. |MinHashEncoder| replace::
     :class:`~skrub.MinHashEncoder`

.. |TargetEncoder| replace::
     :class:`~sklearn.preprocessing.TargetEncoder`

.. |make_pipeline| replace::
     :func:`~sklearn.pipeline.make_pipeline`

.. |Pipeline| replace::
     :class:`~sklearn.pipeline.Pipeline`

.. |HGBC| replace::
     :class:`~sklearn.ensemble.HistGradientBoostingClassifier`

.. |TunedThresholdClassifierCV| replace::
     :class:`~sklearn.model_selection.TunedThresholdClassifierCV`

.. |CalibrationDisplay| replace::
     :class:`~sklearn.calibration.CalibrationDisplay`

.. |pandas.melt| replace::
     :func:`~pandas.melt`

"""

# %%
# The data
# --------
#
# We begin with loading the table from figshare. It has around 100k rows.
from skrub.datasets import fetch_figshare

X = fetch_figshare("48931237").X

# %%
# The total price is the sum of the price per unit of each product in the basket,
# multiplied by their quantity. This will also allow us to define a utility function
# later, in addition of being a useful feature for the learner.
import numpy as np
import pandas as pd

from skrub import TableReport


def total_price(X):
    total_price = pd.Series(np.zeros(X.shape[0]), index=X.index, name="total_price")
    max_item = 24
    for idx in range(1, max_item + 1):
        total_price += X[f"cash_price{idx}"].fillna(0) * X[
            f"Nbr_of_prod_purchas{idx}"
        ].fillna(0)

    return total_price


X["total_price"] = total_price(X)
TableReport(X)

# %%
# Metrics
# -------
#
# To consider the problem from a business perspective, we define our utility function
# by the cost matrix in the function ``credit_gain_score``. False positive and false
# negative predictions incur a negative gain.
#
# Ultimately, we want to maximize this metric. To do so, we can train our learner to
# minimize a proper scoring rule like the log loss.
import sklearn
from sklearn.metrics import log_loss, make_scorer


def credit_gain_score(y_true, y_pred, amount):
    """Define our utility function.

    These numbers are entirely made-up, don't try this at home!
    """
    mask_tn = (y_true == 0) & (y_pred == 0)
    mask_fp = (y_true == 0) & (y_pred == 1)
    mask_fn = (y_true == 1) & (y_pred == 0)

    # Refusing a fraud yields 0 €
    fraudulent_refuse = 0

    # Accepting a fraud costs its whole amount
    fraudulent_accept = -amount[mask_fn].sum()

    # Refusing a legitimate basket transactions cost 5 €
    legitimate_refuse = mask_fp.sum() * -5

    # Accepting a legitimate basket transaction yields 7% of its amount
    legitimate_accept = (amount[mask_tn] * 0.07).sum()

    return fraudulent_refuse + fraudulent_accept + legitimate_refuse + legitimate_accept


def get_results(model, X_test, y_test, threshold, amount, time_to_fit):
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "log_loss": log_loss(y_test, y_proba),
        "gain_score": credit_gain_score(y_test, y_proba > threshold, amount),
        "y_proba": y_proba,
        "y_test": y_test,
        "time_to_fit": time_to_fit,
    }


sklearn.set_config(enable_metadata_routing=True)
gain_score = make_scorer(credit_gain_score).set_score_request(amount=True)

results = dict()

# %%
# Dummy model
# -----------
#
# We first evaluate the performance of a dummy model that always predict the negative
# class (i.e. all transactions are legit).
# This is a good sanity check to make sure our model actually learns something useful.
from time import time

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

target_col = "fraud_flag"
X_ = X.drop(columns=[target_col])
y_ = X[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X_,
    y_,
    test_size=0.1,
    stratify=y_,
    random_state=0,
)

tic = time()
dummy_negative = DummyClassifier(strategy="constant", constant=0).fit(X_train, y_train)
time_to_fit = time() - tic

results["Dummy Negative"] = get_results(
    dummy_negative,
    X_test,
    y_test,
    threshold=0.5,
    amount=X_test["total_price"],
    time_to_fit=time_to_fit,
)

# %%
# Low effort estimator
# --------------------
#
# Next, we use the |TableVectorizer| and a |HGBC| to create a very simple baseline model
# that uses the sparse dataset directly. Note that due to the large number of high
# cardinality columns, we can't use an multi-dimensional encoder like the
# |MinHashEncoder|, because the number of columns would then explode.
#
# Instead, we encode our categories with a |TargetEncoder|.
#
# We also further split the training set into a training and validation set for
# post-training tuning in the post-training phase below.
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder

from skrub import TableVectorizer

X_train_, X_val, y_train_, y_val = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=0
)

low_effort = make_pipeline(
    TableVectorizer(
        high_cardinality=TargetEncoder(),
    ),
    HistGradientBoostingClassifier(),
)

tic = time()
low_effort.fit(X_train_, y_train_)
time_to_fit = time() - tic

# %%
# To maximise our utility function, we have to find the best classification threshold to
# replace the default at 0.5. |TunedThresholdClassifierCV| is a scikit-learn
# meta-estimator that is designed for this exact purpose.
# More details in this `example from scikit-learn <https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html#sphx-glr-auto-examples-model-selection-plot-cost-sensitive-learning-py>`_.
#
# We give it our trained model, and fit it on the validation dataset instead of the
# training dataset to avoid overfitting. Notice that the scoring method is the utility
# function, to which we pass the amount in ``fit``
from sklearn.model_selection import TunedThresholdClassifierCV

low_effort_tuned = TunedThresholdClassifierCV(
    low_effort, cv="prefit", scoring=gain_score, refit=False
).fit(X_val, y_val, amount=X_val["total_price"])

results["Low effort"] = get_results(
    low_effort,
    X_test,
    y_test,
    threshold=low_effort_tuned.best_threshold_,
    amount=X_test["total_price"],
    time_to_fit=time_to_fit,
)

# %%
# We define some plotting functions to display our results.
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.calibration import CalibrationDisplay


def plot_gain_tradeoff(results):
    """Scatter plot of the score gain (y) vs the fit time (x) for each model."""

    rows = []
    for estimator_name, result in results.items():
        result["estimator_name"] = estimator_name
        rows.append(result)
    df = pd.DataFrame(rows)

    names = df["estimator_name"].values
    palette = dict(zip(names, sns.color_palette("colorblind", n_colors=len(names))))

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    sns.scatterplot(
        df,
        x="time_to_fit",
        y="gain_score",
        hue="estimator_name",
        style="estimator_name",
        ax=ax,
        palette=palette,
        s=200,
    )
    ax.grid()

    ticks = df["time_to_fit"].round(3).tolist()
    labels = [f"{tick}s" for tick in ticks]
    ax.set_xticks(ticks, labels)

    ticks = df["gain_score"].round().tolist()
    ticks.insert(1, 650_000)
    labels = [f"{tick:,} €" for tick in ticks]

    ax.set_yticks(ticks, labels)
    ax.set_ylabel("Gain score")
    ax.set_xlabel("Time to fit")
    ax.set_title("Gain score vs Time to fit")
    plt.tight_layout()


def plot_calibration_curve(results):
    """Plot a calibration curve and the log-loss."""

    estimator_names = list(results)
    palette = dict(
        zip(
            estimator_names,
            sns.color_palette("colorblind", n_colors=len(estimator_names)),
        )
    )
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    for name, result in results.items():
        log_loss = str(round(result["log_loss"], 4))
        label = f"{name}, {'log_loss: ' + log_loss}"
        CalibrationDisplay.from_predictions(
            y_true=result["y_test"],
            y_prob=result["y_proba"],
            strategy="quantile",
            label=label,
            ax=ax,
            color=palette[name],
            n_bins=15,
        )
    ax.set_xlim([-0.001, 0.13])
    ax.set_ylim([-0.001, 0.13])
    ax.set_title("Calibration curve")


# %%
# We see below that the low effort classifier significantly improves our gains compared
# to the dummy baseline. The former is of course slower to train than the latter.

plot_gain_tradeoff(results)


# %%
# We also evaluate the calibration of both models. As very few classes are
# positive, we can expect all probabilities to be close to 0. We have to
# zoom on it, and use the "quantile" strategy of |CalibrationDisplay| in order to create
# bins containing an equal number of samples.

plot_calibration_curve(results)


# %%
# Agg-Joiner based estimator
# --------------------------
#
# We first need to split the dataframe between a dataframe representing baskets and a
# dataframe representing products. In other words, we need to revert the join operation
# performed by the creator of this dataset. Conceptually, this is close to a
# |pandas.melt| operation
#
# Note that we don't keep the product ordering information, which is probably not an
# important feature here.


def get_columns_at(idx, cols_2_idx):
    """Small helper that give the position of each of the columns of the idx-th \
        product."""
    cols = [
        "ID",
        target_col,
        f"item{idx}",
        f"cash_price{idx}",
        f"make{idx}",
        f"model{idx}",
        f"goods_code{idx}",
        f"Nbr_of_prod_purchas{idx}",
    ]
    return [cols_2_idx[col] for col in cols]


def melt_multi_columns(X):
    """Create a dataframe where each product is a row."""
    products = []
    cols_2_idx = dict(zip(X.columns, range(X.shape[1])))
    for row in X.values:
        n_products = min(row[cols_2_idx["Nb_of_items"]], 24)
        for idx in range(1, n_products + 1):
            cols = get_columns_at(idx, cols_2_idx)
            products.append(row[cols])

    cols = [
        "ID",
        target_col,
        "item",
        "cash_price",
        "make",
        "model",
        "goods_code",
        "Nbr_of_prod_purchas",
    ]

    products = pd.DataFrame(products, columns=cols)

    for col in ["make", "model"]:
        products[col] = products[col].fillna("None")

    return products


X_train_[target_col] = y_train_
X_val[target_col] = y_val
X_test[target_col] = y_test

baskets_train = X_train_[["ID", "total_price", target_col]]
baskets_val = X_val[["ID", "total_price", target_col]]
baskets_test = X_test[["ID", "total_price", target_col]]

products = melt_multi_columns(X)

TableReport(products)

# %%
# We have to aggregate the products dataframe before joining it back to the basket
# dataframe. Prior to that, we need to apply some preprocessing to deal with
# the high cardinality columns. Since these columns have some morphological variations
# and typos, we use the |MinHashEncoder|.
#
# ``goods_code`` is slightly different, as it represents some merchant IDs, which
# co-occurs for different products. Therefore, we encode it with a |TargetEncoder| as
# we previously did.
#
# To later perform the joiner operation, we must keep the basket ``ID`` with
# ``"passthrough"``.
from skrub import MinHashEncoder


def get_X_y(data):
    return data.drop(columns=[target_col]), data[target_col]


tic = time()
vectorizer = TableVectorizer(
    high_cardinality=MinHashEncoder(),  # applied on ["item", "model", "make"]
    specific_transformers=[
        (TargetEncoder(), ["goods_code"]),
        ("passthrough", ["ID"]),
    ],
)

products_transformed = vectorizer.fit_transform(*get_X_y(products))
time_to_fit = time() - tic

TableReport(products_transformed)

# %%
# Let's now detail how to leverage |AggJoiner| here. We have just encoded each product
# attributes, and now we need to somehow aggregate these product encodings into their
# respective baskets.
#
# By aggregating instead of concatenating, we obtain an invariant number of columns,
# and we remove the sparsity of the dataset.
#
# But which aggregation operation should we choose? Since the |MinHashEncoder| hashes
# ngrams with different hashing functions and return their minimum, it makes sense to
# aggregate different product encodings using their **minimum** for each dimension.
# You can view MinHash minimums as activations.
#
# For numeric columns and columns encoded with the |TargetEncoder|, we take the mean,
# standard deviation, minimum and maximum to extract a representative summary of each
# distribution.
#
# We can apply these two sets of operations by chaining together two |AggJoiner| in
# a |Pipeline| using |make_pipeline|. We also make use of skrub selectors to select
# columns with the ``glob`` syntax.
#
# We need to pass the product dataframe as an auxiliary table argument to AggJoiner
# in ``__init__``. The basket dataframe is our main table, and we pass it during
# ``fit``. We discuss the limitations of this design in the conclusion at the bottom
# of this notebook.
#
# Let's display the output of this preprocessing pipeline.

from sklearn.pipeline import make_pipeline

from skrub import AggJoiner
from skrub import _selectors as s

minhash_cols = "ID" | s.glob("item_*") | s.glob("model_*") | s.glob("make_*")
single_cols = ["ID", "goods_code", "Nbr_of_prod_purchas", "cash_price"]

pipe_agg_joiner = make_pipeline(
    AggJoiner(
        aux_table=s.select(products_transformed, minhash_cols),
        key="ID",
        operations=["min"],
    ),
    AggJoiner(
        aux_table=s.select(products_transformed, single_cols),
        key="ID",
        operations=["mean", "sum", "std", "min", "max"],
    ),
)
basket_train_transformed = pipe_agg_joiner.fit_transform(baskets_train)

TableReport(basket_train_transformed)

# %%
# Now that we get a sense of how the |AggJoiner| can help us, we complete this pipeline
# with a |HGBC| and evaluate our final model.

tic = time()
agg_join_estimator = make_pipeline(
    pipe_agg_joiner,
    HistGradientBoostingClassifier(),
).fit(*get_X_y(baskets_train))
time_to_fit += time() - tic

agg_join_tuned = TunedThresholdClassifierCV(
    agg_join_estimator, cv="prefit", scoring=gain_score, refit=False
).fit(*get_X_y(baskets_val), amount=baskets_val["total_price"])

results["Agg Joiner"] = get_results(
    agg_join_tuned,
    *get_X_y(baskets_test),
    threshold=agg_join_tuned.best_threshold_,
    amount=baskets_test["total_price"],
    time_to_fit=time_to_fit,
)
# %%
# Not only did we improve the gains, but this operation is also much faster than the
# naive low effort!

plot_gain_tradeoff(results)

# %%
# We see that the agg-joiner model is slightly more calibrated, with a lower (better)
# log loss.

plot_calibration_curve(results)

# %%
# Conclusion
# ----------
#
# Many problems involve tables where IDs have a one-to-many relationship. To simplify
# aggregate-then-join operations for machine learning, we can include the |AggJoiner|
# in our pipeline.
#
# One known limitation of both the |AggJoiner| and |Joiner| is that the auxiliary data
# to join is passed during the ``__init__`` method instead of the ``fit`` method, and
# is therefore fixed once the model has been trained.
# This limitation causes two main issues:
#
# 1. **Inefficient model serialization:** Since the dataset has to be pickled along with
# the model, it can result in a massive file size on disk.
#
# 2. **Inflexibility with new, unseen data in a production environment:** To use new
# auxiliary data, you would need to replace the auxiliary table in the AggJoiner that
# was used during ``fit`` with the updated data, which is a rather hacky approach.
#
# These limitations will be addressed later in skrub.
