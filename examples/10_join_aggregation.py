# %%
from skrub.datasets import fetch_figshare
from skrub import TableReport
import pandas as pd

X = fetch_figshare("48931237").X
X["ID"] = X["ID"].astype(str)
TableReport(X)

# %%
from sklearn.metrics import brier_score_loss, log_loss

def get_results(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "log_loss": log_loss(y_test, y_proba),
        "brier_score_loss": brier_score_loss(y_test, y_proba),
    }

results = dict()


# %%

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

target_col = "fraud_flag"
X_ = X.drop(columns=[target_col])
y_ = X[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X_, y_, test_size=0.1, stratify=y_, random_state=0,
)

dummy_negative = DummyClassifier(
    strategy="constant", constant=0
).fit(X_train, y_train)

results["dummy_negative"] = get_results(dummy_negative, X_test, y_test)


# %%
%%time

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from skrub import TableVectorizer


def total_price(X):
    total_price = pd.Series(np.zeros(X.shape[0]), index=X.index, name="total_price")

    for idx in range(1, 24 + 1):
        total_price += X[f"cash_price{idx}"].fillna(0) * X[f"nbr{idx}"].fillna(0)

    return total_price


X_train["total_price"] = total_price(X_train)
X_test["total_price"] = total_price(X_test)

low_effort = make_pipeline(
    TableVectorizer(
        high_cardinality=TargetEncoder(),
        cardinality_threshold=1,
    ),
    HistGradientBoostingClassifier(),
)

low_effort.fit(X_train, y_train)
results["low_effort"] = get_results(low_effort, X_test, y_test)


# %%
import seaborn as sns
from matplotlib import pyplot as plt


def _get_palette(names):
    return dict(zip(names, sns.color_palette("colorblind", n_colors=len(names))))


def plot_metric(results, metric_name):
    """Plot a bar graph comparing all models for the desired metric."""
    values = []
    for named_results in results.values():
        values.append(named_results[metric_name])

    names = list(results)
    palette = _get_palette(names)

    fig, ax = plt.subplots()
    ax = sns.barplot(y=names, x=values, hue=names, palette=palette.values(), orient="h")
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_title(metric_name)
    plt.tight_layout()


plot_metric(results, "log_loss")

# %%
plot_metric(results, "brier_score_loss")

# %%
%%time

def get_columns_at(idx, cols_2_idx):
    cols = [
        "ID", "fraud_flag", f"item{idx}", f"cash_price{idx}", f"make{idx}",
        f"model{idx}", f"goods_code{idx}", f"Nbr_of_prod_purchas{idx}",
    ]
    return [cols_2_idx[col] for col in cols]


def melt_multi_columns(X):
    items = []
    cols_2_idx = dict(zip(X.columns, range(X.shape[1])))
    for row in X.values:
        n_items = min(row[cols_2_idx["Nb_of_items"]], 24)
        for idx in range(1, n_items+1):
            cols = get_columns_at(idx, cols_2_idx)
            items.append(row[cols])
    
    cols = [
        "ID", "fraud_flag", "item", "cash_price", "make",
        "model", "goods_code", "Nbr_of_prod_purchas",
    ]
    return pd.DataFrame(items, columns=cols) 


baskets = X[["ID", target_col]]

items = melt_multi_columns(X)
TableReport(items)

# %%
for col in ["make", "model"]:
    items[col] = items[col].fillna("None")

# %%
%%time

from sklearn.preprocessing import TargetEncoder
from skrub import TableVectorizer, MinHashEncoder

vectorizer = TableVectorizer(
    high_cardinality=MinHashEncoder(),
    specific_transformers=[
        (TargetEncoder(), ["make", "goods_code"]),
        ("passthrough", ["ID"]),
    ]
)
y = items.pop(target_col)
items_transformed = vectorizer.fit_transform(items, y)
TableReport(items_transformed)

# %%

from sklearn.pipeline import make_pipeline
from skrub import AggJoiner
from skrub import _selectors as s

minhash_cols = "ID" | s.glob("item_*") | s.glob("model_*") | s.glob("make_*")
single_cols = ["ID", "goods_code", "Nbr_of_prod_purchas", "cash_price"]

pipe_agg_joiner = make_pipeline(
    AggJoiner(
        aux_table=s.select(items_transformed, minhash_cols),
        key="ID",
        operations=["min"],
    ),
    AggJoiner(
        aux_table=s.select(items_transformed, single_cols),
        key="ID",
        operations=["mean", "sum", "std", "min", "max"],
    )
) 
basket_transformed = pipe_agg_joiner.fit_transform(baskets)

TableReport(basket_transformed)

# %%
