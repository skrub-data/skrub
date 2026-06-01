# %%
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from skrub import SessionEncoder, tabular_pipeline
from skrub.datasets import make_retail_events

# %%
bunch = make_retail_events(n_users=20, n_events=1000, random_state=0)
# %%
X, y = bunch.X, bunch.y
# %%
X.head()
# %%
se = SessionEncoder("timestamp", split_by="user_id", session_gap=30 * 60)
# %%
X_sessions = se.fit_transform(X)
# %%
model = tabular_pipeline("classification")

# %%
splitter = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=splitter, scoring="roc_auc")
print("ROC-AUC without session encoding:", scores.mean())
# ROC-AUC without session encoding: 0.4758557724112403
# %%
scores = cross_val_score(model, X_sessions, y, cv=splitter, scoring="roc_auc")
print("ROC-AUC with session encoding:", scores.mean())
# ROC-AUC with session encoding: 0.48788976843161597

# %%
scores = cross_val_score(
    DummyClassifier(strategy="most_frequent"),
    X_sessions,
    y,
    cv=splitter,
    scoring="roc_auc",
)
print("ROC-AUC with DummyClassifier:", scores.mean())
# ROC-AUC with DummyClassifier: 0.5
# %%

from skrub import SessionEncoder

# Step 1: add session_id
se = SessionEncoder("timestamp", split_by="user_id", session_gap=30 * 60)
X_sessions = se.fit_transform(X)

# Step 2: compute & join session aggregates
session_agg = X_sessions.groupby("timestamp_session_id").agg(
    session_has_add_to_cart=("event_type", lambda x: "add_to_cart" in x.values),
    session_n_events=("event_type", "count"),
    session_mean_price=("price_viewed", "mean"),
    session_dominant_device=("device_type", lambda x: x.mode()[0]),
)
X_enriched = X_sessions.join(session_agg, on="timestamp_session_id")

# Step 3: fit tabular_pipeline on enriched X
model = tabular_pipeline("classification")
# %%
scores = cross_val_score(model, X_enriched, y, cv=splitter, scoring="roc_auc")
print("ROC-AUC with session encoding:", scores.mean())
# %%
