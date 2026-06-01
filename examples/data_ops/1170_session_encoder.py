"""
Use SessionEncoder in DataOps to predict purchases
==================================================
This example shows how to use |SessionEncoder| in a skrub DataOps workflow.
We will:

1. Generate synthetic retail event data
2. Build a baseline classifier on raw event-level features
3. Add session-level and historical features
4. Train the same model again and compare ROC-AUC

The data comes from |make_retail_events| and includes columns such as event type,
device type, viewed price, and timestamp. The target is binary: whether the
session eventually contains a purchase event.
"""

# %%
import skrub
from skrub.datasets import make_retail_events

# %%
events = make_retail_events(n_users=20, n_events=5000, random_state=0)
# %%
# Mark feature and target data with |skrub.X| and |skrub.y| so they can be used
# in a DataOps workflow.

X, y = skrub.X(events.X), skrub.y(events.y)

# %%
# As a sanity check, evaluate a |DummyClassifier| on the original event data
# (without session features).  As it's a DummyClassifier, we expect
# chance-level performance (ROC-AUC of 0.5).
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent")
dummy_pred = X.skb.apply(dummy, y=y)
dummy_learner = dummy_pred.skb.make_learner()
# %%
# Because this is temporal data, we use a time-aware CV strategy.
# We reuse the same splitter for all evaluations.
from sklearn.model_selection import TimeSeriesSplit

splitter = TimeSeriesSplit(n_splits=5)
dummy_results = skrub.cross_validate(
    dummy_learner, environment=dummy_pred.skb.get_data(), cv=splitter, scoring="roc_auc"
)
print(f"ROC-AUC with DummyClassifier: {dummy_results['test_score'].mean():.3f}")

# %%
# Try a real model with |tabular_pipeline|, first on raw event-level data.
from skrub import tabular_pipeline

model = tabular_pipeline("classification")

pred = X.skb.apply(model, y=y)
learner = pred.skb.make_learner()
results = skrub.cross_validate(
    learner, environment=pred.skb.get_data(), cv=splitter, scoring="roc_auc"
)
print(f"ROC-AUC without session encoding: {results['test_score'].mean():.3f}")

# %%
# This baseline is limited because it cannot directly use session-level behavior
# (for example, whether "add_to_cart" happened in the same session).
#
# Next, create sessions with |SessionEncoder|. We define boundaries from
# ``timestamp`` within each ``user_id``. A new session starts after more than
# 30 minutes of inactivity (``session_gap`` is in seconds).
# %%
from skrub import SessionEncoder

se = SessionEncoder("timestamp", split_by="user_id", session_gap=30 * 60)
X_sessions = X.skb.apply(se)

# %%
# ``timestamp_session_id`` identifies the session of each event.
# We use it to compute session-level aggregates and join them back to event-level rows.
#
# We will compute the following session-level features:
# - ``session_has_add_to_cart``: whether the session includes at least one "add_to_cart"
#   event
# - ``session_n_events``: the total number of events in the session
# - ``session_mean_price``: the mean price viewed during the session
# - ``session_dominant_device``: the most frequently used device type in the session
# - ``event_rank_in_session``: the rank of the event within its session (0 for the
#   first event, 1 for the second, etc.)
# - ``is_last_event_in_session``: whether the event is the last event in its session
#
# We also compute one user-level historical feature after sorting by timestamp:
# - ``time_since_last_event``: the time in seconds since the previous event for the
#   same user (NaN for the first event of each user)


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
    grouped = df.groupby("timestamp_session_id")
    df["event_rank_in_session"] = grouped.cumcount()
    session_sizes = grouped["event_type"].transform("size")
    df["is_last_event_in_session"] = df["event_rank_in_session"].eq(session_sizes - 1)
    return df


def compute_historical_features(df):
    # Preserve input row order after timestamp-based computations.
    df["_row_order"] = df.index
    df = df.sort_values("timestamp")
    df["time_since_last_event"] = (
        df.groupby("user_id")["timestamp"].diff().dt.total_seconds()
    )
    df = df.sort_values("_row_order").drop(columns="_row_order")
    return df


X_enriched = X_sessions.skb.apply_func(compute_session_features)
X_enriched = X_enriched.skb.apply_func(compute_historical_features)
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
# The enriched model should outperform the baseline, showing the value of
# session-level context for conversion prediction.
#
# In DataOps, these aggregations are evaluated with temporal ordering in mind,
# which helps prevent leakage: features for an event are computed only from data
# available up to that event timestamp.
#
# This example focuses on SessionEncoder usage, so we intentionally keep modeling
# simple (no hyperparameter tuning and only a small set of engineered features).
