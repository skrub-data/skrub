"""
Self-aggregation with MovieLens
===============================

MovieLens is a famous movie dataset used for both explicit
and implicit recommender system. It provides a main table,
"ratings", that can be viewed as logs or transactions, comprised
of only 4 columns: userId, movieId, rating and timestamp.
MovieLens also gives a contextual table "movies", including
movieId, title and types, to enable content-based feature extraction.

In this notebook, we only deal with the main table "ratings".
Our objective is **not to achieve state-of-the-art performance** on
the explicit regression task, but rather to illustrate how to perform
feature engineering in a simple way using |AggJoiner| and |AggTarget|.
Note that our performance is higher than the baseline of using the mean
rating per movies.


.. |AggJoiner| replace::
     :class:`~skrub.AggJoiner`

.. |AggTarget| replace::
     :class:`~skrub.AggTarget`

.. |TableVectorizer| replace::
     :class:`~skrub.TableVectorizer`

.. |DatetimeEncoder| replace::
     :class:`~skrub.DatetimeEncoder`

.. |TargetEncoder| replace::
     :class:`~skrub.TargetEncoder`

.. |Pipeline| replace::
     :class:`~sklearn.pipeline.Pipeline`

.. |RandomizedSearchCV| replace::
     :class:`~sklearn.model_selection.RandomizedSearchCV`

.. |TimeSeriesSplit| replace::
     :class:`~sklearn.model_selection.TimeSeriesSplit`

.. |HGBR| replace::
     :class:`~sklearn.ensemble.HistGradientBoostingRegressor`
"""

###############################################################################
# The data
# --------
#
# We begin with loading the ratings table from MovieLens.
# Note that we use the light version (100k rows).
import pandas as pd
import numpy as np

from skrub.datasets import fetch_movielens


ratings = fetch_movielens(dataset_id="ratings")
ratings = ratings.X.sort_values("timestamp").reset_index(drop=True)
ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")

X = ratings[["userId", "movieId", "timestamp"]]
y = ratings["rating"]
print(X.shape)
print(X.head())

###############################################################################
# Encoding the timestamp with a TableVectorizer
# ---------------------------------------------
#
# Our first step is to extract features from the timestamp, using the
# |TableVectorizer|. Natively, it uses the |DatetimeEncoder| on datetime
# columns, and doesn't interact with numerical columns.
from skrub import TableVectorizer, DatetimeEncoder


table_vectorizer = TableVectorizer(
    datetime_transformer=DatetimeEncoder(add_day_of_the_week=True)
)
table_vectorizer.set_output(transform="pandas")
X_date_encoded = table_vectorizer.fit_transform(X)
print(X_date_encoded)

###############################################################################
# We can now make a couple of plots and gain some insight on our dataset.
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


def make_barplot(x, y, title):
    norm = plt.Normalize(y.min(), y.max())
    cmap = plt.get_cmap("magma")

    ax = sns.barplot(
        x=x,
        y=y,
        palette=cmap(norm(y)),
    )
    plt.xticks(rotation=30)
    plt.xlabel(title)
    plt.ylabel(None)
    plt.tight_layout()


# O is Monday, 6 is Sunday

daily_volume = X_date_encoded["timestamp_dayofweek"].value_counts().sort_index()

make_barplot(
    x=daily_volume.index, y=daily_volume.values, title="Daily volume of ratings"
)

###############################################################################
# We also display the distribution of our target ``y``.
rating_count = y.value_counts().sort_index()

make_barplot(
    x=rating_count.index,
    y=rating_count.values,
    title="Distribution of ratings given to movies",
)

###############################################################################
# AggJoiner: aggregate auxiliary tables, then join
# ------------------------------------------------
#
# We now want to extract aggregated datetime features both about the users
# and the movies. These features answer questions like
# *"How many times has this user rated a movie in 2008?"* or
# *"What is the most frequent hour for this movie to be rated?"*.
#
# Below, |AggJoiner| first performs the aggregations defined by ``agg_ops``,
# on ``"userId"``, then on ``"movieId"``, before joining both results on
# the main table.
# The ``tables`` tuple specifies:
#
# - the auxiliary tables on which to perform the aggregation
# - the key used to group and join
# - the columns to aggregate (``timestamp_cols``).
#
# Here, the auxiliary tables are the main table ``X_date_encoded`` itself.
#
# The ``main_key`` indicates the key of the main table to join each tuple of
# ``tables``.
from skrub import AggJoiner


timestamp_cols = [
    "timestamp_year",
    "timestamp_month",
    "timestamp_hour",
    "timestamp_dayofweek",
]

agg_joiner = AggJoiner(
    tables=[
        (X_date_encoded, "userId", timestamp_cols),
        (X_date_encoded, "movieId", timestamp_cols),
    ],
    main_key=["userId", "movieId"],
    suffixes=["_user", "_movie"],
    agg_ops=["value_counts"],
)
X_transformed = agg_joiner.fit_transform(X_date_encoded)

print(X_transformed.shape)
print(X_transformed.head())

###############################################################################
# AggTarget: aggregate y, then join
# ---------------------------------
#
# We just expanded our timestamp to create datetime features.
#
# Let's now perform a similar expansion for the target ``y``. Of course,
# the biggest risk of doing target expansion with multiple pandas operations
# is to end up leaking the target.
#
# Similarly to |AggJoiner|, the |AggTarget| transformer allows you to
# aggregate the target ``y`` before joining it on the main table, without
# risk of leaking.
#
# You can think of it as a generalization of the |TargetEncoder|, which
# encodes categorical features based on the target.
#
# Here, we group the target by ``"userId"``, then by ``"movieId"``, and for
# each group we compute the histogram of the target with 3 bins.
#
# Finally, we join both user and movie aggregations back on the main table.
from skrub import AggTarget


agg_target = AggTarget(
    main_key=["userId", "movieId"],
    suffixes=["_user", "_movie"],
    agg_ops=["hist(3)"],
)
X_transformed = agg_target.fit_transform(X, y)

print(X_transformed.shape)
print(X_transformed.head())

###############################################################################
# Chaining everything together in a pipeline
# ------------------------------------------
#
# To perform cross-validation and enable hyper-parameter tuning, we gather
# all elements into a scikit-learn |Pipeline| and use scikit-learn |HGBR|.
#
# Since the auxiliary tables of |AggJoiner| are the main table itself, we need
# to use the output of the previous layer (here the |TableVectorizer|).
# We enable this behaviour by **using the placeholder** ``"X"`` **instead of a
# dataframe** in the ``tables`` tuple.
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline


table_vectorizer = TableVectorizer(
    datetime_transformer=DatetimeEncoder(add_day_of_the_week=True)
)
table_vectorizer.set_output(transform="pandas")

agg_joiner = AggJoiner(
    tables=[
        ("X", "userId", timestamp_cols),
        ("X", "movieId", timestamp_cols),
    ],
    main_key=["userId", "movieId"],
    suffixes=["_user", "_movie"],
    agg_ops=["value_counts"],
)

agg_target = AggTarget(
    main_key=["userId", "movieId"],
    suffixes=["_user", "_movie"],
    agg_ops=["value_counts"],
)

pipeline = make_pipeline(
    table_vectorizer,
    agg_joiner,
    agg_target,
    HistGradientBoostingRegressor(learning_rate=0.1, max_depth=4, max_iter=40),
)

pipeline

###############################################################################
# Hyper-parameters tuning and cross validation
# --------------------------------------------
#
# We can finally create our hyper-parameter search space, and use a
# |RandomizedSearchCV|. We select the cross validation splitter to be
# the |TimeSeriesSplit| to prevent leakage, since our data are timestamped
# logs.
#
# The score used in this regression task is the R2. Remember that the R2
# evaluates the relative performance compared to the naive baseline consisting
# in always predicting the mean value of ``y_test``.
# Therefore, the R2 is 0 when ``y_pred = y_true.mean()`` and is upper bounded
# to 1 when ``y_pred = y_true``.
list(pipeline.named_steps)

###############################################################################
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


params = {
    "aggjoiner__agg_ops": ["value_counts"],
    "aggtarget__agg_ops": ["mean", "hist(3)", "hist(5)", "hist(7)", "value_counts"],
}

cv = GridSearchCV(pipeline, params, cv=TimeSeriesSplit(n_splits=10))
cv.fit(X, y)

results = pd.DataFrame(
    np.vstack([cv.cv_results_[f"split{idx}_test_score"] for idx in range(10)]),
    columns=cv.cv_results_["param_aggtarget__agg_ops"],
)

###############################################################################
# To get a better sense of the learning performances of our simple pipeline,
# we also compute the average rating of each movie in the training set,
# and uses this average to predict the ratings in the test set.
from sklearn.metrics import r2_score


def baseline_r2(X, y, train_idx, test_idx):
    """Compute the average rating for all movies in the train set,
    and map these averages to the test set as a prediction.

    If a movie in the test set is not present in the training set,
    we simply predict the global average rating of the training set.
    """
    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    X_train["y"] = y_train

    movie_avg_rating = X_train.groupby("movieId")["y"].mean().to_frame().reset_index()

    y_pred = X_test.merge(movie_avg_rating, on="movieId", how="left")["y"]
    y_pred = y_pred.fillna(y_pred.mean())

    return r2_score(y_true=y_test, y_pred=y_pred)


all_baseline_r2 = []
for train_idx, test_idx in TimeSeriesSplit(n_splits=10).split(X, y):
    all_baseline_r2.append(baseline_r2(X, y, train_idx, test_idx))

results.insert(0, "naive mean estimator", all_baseline_r2)

# we only keep the 5 out of 10 last results
# because the initial size of the train set is rather small
sns.boxplot(results.tail(5))
plt.tight_layout()

###############################################################################
# The naive estimator has a lower performance than our pipeline, which means
# that our extracted features brought some predictive power.
#
# It seems that using the ``"value_counts"`` as an aggregation operator for
# |AggTarget| yields better performances than using the mean (which is
# equivalent to the |TargetEncoder|).
#
# Here, the number of bins to encode the target is proportional to the
# performance: computing the mean yields a single statistic, histograms yields
# a density over a reduced set of bins and ``"value_counts"`` yields an
# exhaustive histogram over all the possible values of ratings
# (here 10 different values, from 0.5 to 5).
