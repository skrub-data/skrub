"""

Sessions in time-based data: Predicting user purchases with the SessionEncoder
===============================================================================

.. |SessionEncoder| replace:: :class:`~skrub.SessionEncoder`
.. |make_retail_events| replace:: :func:`~skrub.datasets.make_retail_events`
.. |tabular_pipeline| replace:: :func:`~skrub.tabular_pipeline`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |DummyClassifier| replace:: :class:`~sklearn.dummy.DummyClassifier`
.. |TimeSeriesSplit| replace:: :class:`~sklearn.model_selection.TimeSeriesSplit`
.. |BaseEstimator| replace:: :class:`~sklearn.base.BaseEstimator`
.. |TransformerMixin| replace:: :class:`~sklearn.base.TransformerMixin`

This example shows how to use |SessionEncoder| in a scikit-learn pipeline to
create session-level features (sessionization) for conversion prediction, that is
predicting whether a user session will eventually lead to a purchase.

.. topic:: What is sessionization?

    Sessionization is the process of grouping a sequence of events (like user
    interactions) into meaningful sessions. A session typically starts fresh or
    after a period of inactivity. For example, in an online retail context, you
    might define a new session whenever more than 30 minutes pass with no activity
    from a user. This allows you to extract session-level features (like the total
    number of events in a session or the dominant device type used) which often have
    greater predictive power than raw individual events.

We will:

1. Use |make_retail_events| to generate synthetic retail event data
2. Build a baseline classifier on raw event-level features with the |tabular_pipeline|
3. Add session-level and historical features with |SessionEncoder|
4. Train the same model again and compare ROC-AUC

The data includes columns such as event type, device type, viewed price, and
timestamp. The target is binary: whether the session eventually contains a
purchase event or not.

"""

# %%
# Since this is temporal data, we use a time-aware CV strategy with
# |TimeSeriesSplit| to avoid leakage. We reuse the same splitter for all evaluations.
from sklearn.model_selection import TimeSeriesSplit

splitter = TimeSeriesSplit(n_splits=5)
# %%
# We begin by generating the data with |make_retail_events| and defining out
# features and target.
from skrub import TableReport
from skrub.datasets import make_retail_events

events = make_retail_events(n_users=20, n_events=5000, random_state=0)
X, y = events.X, events.y
TableReport(X)
# %%
# The data contains 5000 events from 20 users, where each event is timestamped.
# Other columns include the event type, device used by the user, page category,
# time spent on page and price of the item. The target variable indicates whether
# a user session eventually contains a purchase event: all events in that session
# will have a target value of 1 if a purchase happens, and 0 otherwise.

# %%
# Sanity check: evaluate a DummyClassifier on raw event data
# ---------------------------------------------------------------
# We begin by evaluating a |DummyClassifier| on the original event data
# (without session features).  Since it's a |DummyClassifier|, we expect
# chance-level performance (ROC-AUC of 0.5).
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

dummy = DummyClassifier(strategy="most_frequent")

scores = cross_val_score(dummy, X, y, cv=splitter, scoring="roc_auc")
print(f"ROC-AUC with DummyClassifier: {scores.mean():.3f}")

# %%
# First attempt: training a model without using session-level features
# --------------------------------------------------------------------
# We first use the |tabular_pipeline| on raw event-level data, without any session
# encoding or aggregation. This serves as a baseline to compare against the enriched
# model later.
# Remember that the |tabular_pipeline| will automatically add a |TableVectorizer|
# to perform feature engineering, so the model can still learn from the raw event
# features. However, it won't be able to directly capture session-level patterns.
from skrub import tabular_pipeline

model = tabular_pipeline("classification")

scores = cross_val_score(model, X, y, cv=splitter, scoring="roc_auc")
print(f"ROC-AUC without session encoding: {scores.mean():.3f}")
# %%
# The model is not performing much better than the DummyClassifier, which suggests
# that raw event-level features are not sufficient for good conversion prediction.
# This baseline is limited because it cannot directly use session-level behavior
# (for example, whether "add_to_cart" happened in the same session).

# %%
# A better approach: session encoding and aggregation
# ------------------------------------------------------
# Next, we use the |SessionEncoder| to create session-level features that we can
# aggregate over. We define a session boundary as "a user has been inactive for
# more than 30 minutes". The |SessionEncoder| will create a new column
# ``timestamp_session_id`` that assigns a unique session ID to each session detected.
# The parameter ``session_gap=30 * 60`` specifies the inactivity threshold in
# seconds (30 minutes).
#
# Note that session-based features involve aggregations, which must be performed
# only on the training data within each fold to avoid leakage. In a scikit-learn
# pipeline, we can achieve this by using |SessionEncoder| followed by a custom
# transformer that computes session aggregates, and ensuring that the pipeline is
# properly fitted within each fold of cross-validation.
# %%
from skrub import SessionEncoder, tabular_pipeline

se = SessionEncoder("timestamp", split_by="user_id", session_gap=30 * 60)
# Here we fit the SessionEncoder on the entire dataset for demonstration purposes
X_sessions = se.fit_transform(X)
X_sessions.head()

# %%
# Defining a custom transformer for session-level aggregation
# -----------------------------------------------------------
# To avoid data leakage and maintain a clean pipeline, we can create a custom
# transformer that inherits from |BaseEstimator| and |TransformerMixin| and
# computes session-level aggregates within a scikit-learn pipeline.
# This transformer will be fitted and applied separately within each fold of
# cross-validation, ensuring that session features are computed only on the training
# data of each fold.

from sklearn.base import BaseEstimator, TransformerMixin


class SessionAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Compute session-level aggregates
        session_agg = X.groupby("timestamp_session_id").agg(
            session_has_add_to_cart=("event_type", lambda x: "add_to_cart" in x.values),
            session_n_events=("event_type", "count"),
            session_mean_price=("price_viewed", "mean"),
            session_dominant_device=("device_type", lambda x: x.mode()[0]),
        )
        # Join back to the original data
        return X.join(session_agg, on="timestamp_session_id")


# %%
# Then, we create a pipeline that includes the |SessionEncoder|, our custom
# ``SessionAggregator``, and the |tabular_pipeline| for classification. This
# pipeline will be used in cross-validation to evaluate the model
# with session features.
from sklearn.pipeline import make_pipeline

model = make_pipeline(se, SessionAggregator(), tabular_pipeline("classification"))
scores = cross_val_score(model, X, y, cv=splitter, scoring="roc_auc")
print("ROC-AUC with session encoding:", scores.mean())

# %%
# As expected, the model with session encoding performs much better than the baseline
# without session features, demonstrating the value of sessionization for conversion
# prediction.
#
# The fact that we are working with aggregation means that it was necessary to
# create a custom transformer to compute session-level features. This situation
# can be avoided by using the skrub DataOps workflow, which allows for more
# flexible data transformations without needing to fit everything within a
# scikit-learn pipeline.
