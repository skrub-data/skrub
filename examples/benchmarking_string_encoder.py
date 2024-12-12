# %%
# Benchmarking different parameters for the StringEncoder transformer

# %%
from skrub.datasets import fetch_toxicity

dataset = fetch_toxicity()
X, y = dataset.X, dataset.y
X["is_toxic"] = y

y = X.pop("is_toxic").map({"Toxic": 1, "Not Toxic": 0})

# %%
from skrub import TableReport

TableReport(X)

# %%
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

from skrub import TableVectorizer


def plot_box_results(named_results):
    fig, ax = plt.subplots()
    names, scores = zip(
        *[(name, result["test_score"]) for name, result in named_results]
    )
    ax.boxplot(scores)
    ax.set_xticks(range(1, len(names) + 1), labels=list(names), size=12)
    ax.set_ylabel("ROC AUC", size=14)
    plt.title(
        "AUC distribution across folds (higher is better)",
        size=14,
    )
    plt.show()


def plot_performance_tradeoff(results):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    # markers = ["s", "o", "^", "x"]
    for idx, (name, result) in enumerate(results):
        ax.scatter(
            result["fit_time"],
            result["test_score"],
            label=name,
            # marker=markers[idx],
        )
        mean_fit_time = np.mean(result["fit_time"])
        mean_score = np.mean(result["test_score"])
        ax.scatter(
            mean_fit_time,
            mean_score,
            color="k",
            # marker=markers[idx],
        )
        std_fit_time = np.std(result["fit_time"])
        std_score = np.std(result["test_score"])
        ax.errorbar(
            x=mean_fit_time,
            y=mean_score,
            yerr=std_score,
            fmt="none",
            c="k",
            capsize=2,
        )
        ax.errorbar(
            x=mean_fit_time,
            y=mean_score,
            xerr=std_fit_time,
            fmt="none",
            c="k",
            capsize=2,
        )

        ax.set_xlabel("Time to fit (seconds)")
        ax.set_ylabel("ROC AUC")
        ax.set_title("Prediction performance / training time trade-off")

    ax.annotate(
        "",
        xy=(1.5, 0.98),
        xytext=(8.5, 0.90),
        arrowprops=dict(arrowstyle="->", mutation_scale=15),
    )
    # ax.text(8, 0.86, "Best time / \nperformance trade-off")
    ax.legend(bbox_to_anchor=(1, 0.3))
    plt.show()


# %%
from skrub import StringEncoder

results = []

# %%
default_pipe = make_pipeline(
    TableVectorizer(high_cardinality=StringEncoder(n_components=30)),
    HistGradientBoostingClassifier(),
)
gap_results = cross_validate(default_pipe, X, y, scoring="roc_auc")
results.append(("tfidf_default", gap_results))

plot_box_results(results)

# %%
hashing_pipe = make_pipeline(
    TableVectorizer(high_cardinality=StringEncoder(n_components=30)),
    HistGradientBoostingClassifier(),
)
results_ = cross_validate(hashing_pipe, X, y, scoring="roc_auc")
results.append(("hashing_default", results_))

plot_box_results(results)

# %%
configurations = {
    "ngram_range": [(1, 1), (3, 4)],
    "analyzer": ["word", "char", "char_wb"],
    "vectorizer": ["tfidf"],
    "n_components": [30],
    # "tf_idf_followup": [True],
}

# %%
from sklearn.model_selection import ParameterGrid

config_grid = ParameterGrid(configurations)

import polars as pl
from tqdm import tqdm


# %%
def format_name(params):
    s = (
        f'{params["vectorizer"]},'
        + f'{params["ngram_range"]},'
        + f'{params["analyzer"]},'
        + f'{params["tf_idf_followup"]}'
    )
    return s


results = []


for params in tqdm(config_grid, total=len(config_grid)):
    print(params)
    this_pipe = make_pipeline(
        TableVectorizer(high_cardinality=StringEncoder(**params)),
        HistGradientBoostingClassifier(),
    )
    results_ = cross_validate(this_pipe, X, y, scoring="roc_auc")
    print(results_)
    params.update(
        {
            "fit_time": list(results_["fit_time"]),
            "test_score": list(results_["test_score"]),
            "ngram_range": str(params["ngram_range"]),
        }
    )
    results.append(params)

df = pl.from_dicts(results)

# %%
df = df.with_columns(
    mean_fit_time=pl.col("fit_time").list.mean(),
    mean_score=pl.col("test_score").list.mean(),
    std_fit_time=pl.col("fit_time").list.std(),
    std_score=pl.col("test_score").list.std(),
)

# %%
plot_performance_tradeoff(results)

# %%

# %%
import pandas as pd
import seaborn as sns


def pareto_frontier_plot(
    data,
    x_var,
    y_var,
    hue_var,
    # palette,
    # hue_order,
    ax,
    ax_title=None,
    ax_xlabel="",
):
    if not isinstance(data, pd.DataFrame):
        raise ValueError()
    x = data[x_var]
    y = data[y_var]

    # ax.set_xscale("log")

    xs = np.array(x)
    ys = np.array(y)
    perm = np.argsort(xs)
    xs = xs[perm]
    ys = ys[perm]

    sns.scatterplot(
        data=data,
        x=x_var,
        y=y_var,
        hue=hue_var,
        ax=ax,
        palette="tab10",
        # hue_order=hue_order,
    )

    # for row in df.iter_rows(named=True):
    #     mean_fit_time = row["mean_fit_time"]
    #     mean_score = row["mean_score"]
    #     std_fit_time = row["std_fit_time"]
    #     std_score = row["std_score"]

    #     ax.errorbar(mean_fit_time, mean_score, std_fit_time, std_score, c="k")

    xs_pareto = [xs[0], xs[0]]
    ys_pareto = [ys[0], ys[0]]
    for i in range(1, len(xs)):
        if ys[i] > ys_pareto[-1]:
            xs_pareto.append(xs[i])
            ys_pareto.append(ys_pareto[-1])
            xs_pareto.append(xs[i])
            ys_pareto.append(ys[i])
    xs_pareto.append(ax.get_xlim()[1])
    ys_pareto.append(ys_pareto[-1])

    ax.plot(xs_pareto, ys_pareto, "--", color="k", linewidth=2, zorder=0.8)
    ax.set_ylabel("")
    # ax.set_title(ax_title)
    h, l = ax.get_legend_handles_labels()
    # ax.legend(
    #     h,
    #     [constants.LABEL_MAPPING[hue_var][_] for _ in l],
    #     title=None,
    # )
    ax.set_xlabel(ax_xlabel)

    # ax.set_ylim([-0.5, 0.6])
    # ax.axhspan(0, -0.5, zorder=0, alpha=0.05, color="red")

    optimal_y = ys_pareto[-1]
    return (h, l), optimal_y


# %%
fig, axs = plt.subplots(1, 3, figsize=(10, 3))

for ax, hue_var in zip(axs, ["analyzer", "ngram_range", "vectorizer"]):
    pareto_frontier_plot(
        df.to_pandas(),
        x_var="mean_fit_time",
        y_var="mean_score",
        hue_var=hue_var,
        ax=ax,
    )

# %%
