# %%
# Benchmarking different parameters for the StringEncoder transformer
# This script is used to test different parameters to use with the StringEncoder
# and see which configurations work best.
#
# For the moment, I am only considering the Toxicity dataset to test the performance,
# and more tables should be tested to have more reliable results. It's still a
# good start.
#
# The version of the StringEncoder used here will be simplified for the next release.

# %%
# Import all the required libraries
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

from skrub import StringEncoder, TableVectorizer

# %%
# Import the toxicity dataset and prepare it for the experiments.
from skrub.datasets import fetch_toxicity

dataset = fetch_toxicity()
X, y = dataset.X, dataset.y
X["is_toxic"] = y

y = X.pop("is_toxic").map({"Toxic": 1, "Not Toxic": 0})

from skrub import TableReport

TableReport(X)

# %%
# Prepare the parameter grid to evaluate.
from sklearn.model_selection import ParameterGrid

configurations = {
    "ngram_range": [(1, 1), (1, 2), (3, 4)],
    "analyzer": ["word", "char", "char_wb"],
    "vectorizer": ["tfidf", "hashing"],
    "n_components": [30],
    "tf_idf_followup": [True, False],
}

config_grid = ParameterGrid(configurations)


# %%
def format_name(params):
    # Simple helper function to format the labels
    s = (
        f'{params["vectorizer"]},'
        + f'{params["ngram_range"]},'
        + f'{params["analyzer"]},'
        + f'{params["tf_idf_followup"]}'
    )
    return s


# %%
# Run the experiments and save all the results in a dataframe.

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

df.write_parquet("results.parquet")


# %%
# Build the Pareto frontier plot for a given set of variables, and color the
# dots by a specific variable.
def pareto_frontier_plot(
    data,
    x_var,
    y_var,
    hue_var,
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
    )

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
    ax.set_xscale("log")
    h, l = ax.get_legend_handles_labels()
    ax.set_xlabel(ax_xlabel)

    return (h, l)


# %%
# Use the function defined above to plot three different Pareto plots that are
# colored by hue_var.
fig, axs = plt.subplots(1, 3, figsize=(10, 3))

for ax, hue_var in zip(axs, ["analyzer", "ngram_range", "vectorizer"]):
    pareto_frontier_plot(
        df.to_pandas(),
        x_var="mean_fit_time",
        y_var="mean_score",
        hue_var=hue_var,
        ax=ax,
    )
fig.savefig("results.png")
# %%
# Boxplots comparing the test score for different analyzers
sns.catplot(
    data=df.to_pandas(),
    x="analyzer",
    y="test_score",
    hue="ngram_range",
    kind="box",
    col="vectorizer",
)
# %%
g = sns.catplot(
    data=df.to_pandas(),
    x="analyzer",
    y="fit_time",
    hue="ngram_range",
    kind="box",
    col="vectorizer",
)
g.set(ylim=(0, 30))
# %%
# From the results, it's clear that the tfidf vectorizer is much faster than the
# hashing vectorizer, and achieves similar if not better test score. The best
# ngram_range is (3,4), and `char` and `char_wb` are the better analyzers.
#
# As mentioned before, this is a preliminary study, but it's already providing
# interesting results and an indication of what should be used as default
# parameters for the StringEncoder.
