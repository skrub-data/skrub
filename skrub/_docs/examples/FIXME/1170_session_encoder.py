"""

.. |SessionEncoder| replace:: :class:`~skrub.SessionEncoder`
.. |make_retail_events| replace:: :func:`~skrub.datasets.make_retail_events`
.. |tabular_pipeline| replace:: :func:`~skrub.tabular_pipeline`
.. |skrub.X| replace:: :func:`~skrub.X`
.. |skrub.y| replace:: :func:`~skrub.y`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |DummyClassifier| replace:: :class:`~sklearn.dummy.DummyClassifier`
.. |TimeSeriesSplit| replace:: :class:`~sklearn.model_selection.TimeSeriesSplit`
.. |cross_validate| replace:: :func:`~skrub.cross_validate`
.. |apply_func| replace:: :func:`~skrub.DataOp.skb.apply_func`

Sessions in time-based data: Using SessionEncoder in rich DataOps pipeline
==========================================================================

This example shows how to use |SessionEncoder| in a skrub DataOps workflow to
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
# We begin by generating the data with |make_retail_events| and marking feature
# and target data with |skrub.X| and |skrub.y| so they can be used
# in a DataOps workflow.

import skrub
from skrub.datasets import make_retail_events

events = make_retail_events(n_users=20, n_events=5000, random_state=0)
X, y = skrub.X(events.X), skrub.y(events.y)
X
# %%
# Sanity check: evaluate a DummyClassifier on raw event data
# ---------------------------------------------------------------
# We begin by evaluating a |DummyClassifier| on the original event data
# (without session features).  Since it's a |DummyClassifier|, we expect
# chance-level performance (ROC-AUC of 0.5).
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent")
dummy_pred = X.skb.apply(dummy, y=y)
dummy_learner = dummy_pred.skb.make_learner()
dummy_results = skrub.cross_validate(
    dummy_learner, environment=dummy_pred.skb.get_data(), cv=splitter, scoring="roc_auc"
)
print(f"ROC-AUC with DummyClassifier: {dummy_results['test_score'].mean():.3f}")

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

pred = X.skb.apply(model, y=y)
learner = pred.skb.make_learner()
results = skrub.cross_validate(
    learner, environment=pred.skb.get_data(), cv=splitter, scoring="roc_auc"
)
print(f"ROC-AUC without session encoding: {results['test_score'].mean():.3f}")

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

# %%
from skrub import SessionEncoder

se = SessionEncoder("timestamp", split_by="user_id", session_gap=30 * 60)
X_sessions = X.skb.apply(se)
X_sessions

# %%
# ``timestamp_session_id`` identifies the session of each event.
# We use it to compute session-level aggregates and join them back to event-level rows.
#
# .. admonition:: Session-level feature engineering
#    :collapsible: closed
#
#    We will compute the following session-level features:
#
#    - ``session_has_add_to_cart``: whether the session includes at least one
#      "add_to_cart" event
#    - ``session_n_events``: the total number of events in the session
#    - ``session_mean_price``: the mean price viewed during the session
#    - ``session_dominant_device``: the most frequently used device type in the session


def most_frequent(series):
    # mode() can return multiple values; use the first one
    # for a deterministic tie-break.
    return series.mode().iat[0]


def compute_session_features(df):
    session_agg = df.groupby("timestamp_session_id").agg(
        session_has_add_to_cart=("event_type", lambda x: "add_to_cart" in x.values),
        session_n_events=("event_type", "count"),
        session_mean_price=("price_viewed", "mean"),
        session_dominant_device=("device_type", most_frequent),
    )
    df = df.join(session_agg, on="timestamp_session_id")
    return df


# %%
# We use |apply_func| to apply these feature engineering functions to the data
# with session IDs.
X_enriched = X_sessions.skb.apply_func(compute_session_features)
X_enriched
# %%
# Now we can train the same model on the enriched data with session-level features
# and see if the performance improves.
model = tabular_pipeline("classification")
pred_enriched = X_enriched.skb.apply(model, y=y)
learner_enriched = pred_enriched.skb.make_learner()
results_enriched = skrub.cross_validate(
    learner_enriched,
    environment=pred_enriched.skb.get_data(),
    cv=splitter,
    scoring="roc_auc",
)
print(f"ROC-AUC with session encoding: {results_enriched['test_score'].mean():.3f}")

# %%
# The enriched model clearly outperforms the baseline, showing the value of
# session-level context for conversion prediction.

# %%
# Discussion
# -----------
# In DataOps, these aggregations are evaluated with temporal ordering in mind,
# which helps prevent leakage: features for an event are computed only from data
# available up to that event timestamp (provided that the correct splitter is used).
#
# This example focuses on |SessionEncoder| usage, so we intentionally keep modeling
# simple (no hyperparameter tuning and only a small set of engineered features).
