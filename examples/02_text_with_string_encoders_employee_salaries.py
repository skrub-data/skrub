"""
.. _example_string_encoders:

=====================================================
Various string encoders: a sentiment analysis example
=====================================================

In this example, we explore the performance of string and categorical encoders
available in skrub.

.. |GapEncoder| replace::
     :class:`~skrub.GapEncoder`

.. |MinHashEncoder| replace::
     :class:`~skrub.MinHashEncoder`

.. |TextEncoder| replace::
     :class:`~skrub.TextEncoder`

.. |StringEncoder| replace::
     :class:`~skrub.StringEncoder`

.. |TableReport| replace::
     :class:`~skrub.TableReport`

.. |TableVectorizer| replace::
     :class:`~skrub.TableVectorizer`

.. |pipeline| replace::
     :class:`~sklearn.pipeline.Pipeline`

.. |HistGradientBoostingRegressor| replace::
     :class:`~sklearn.ensemble.HistGradientBoostingRegressor`

.. |RandomizedSearchCV| replace::
     :class:`~sklearn.model_selection.RandomizedSearchCV`

.. |GridSearchCV| replace::
     :class:`~sklearn.model_selection.GridSearchCV`
"""

# %%
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
X, y = dataset.X, dataset.y


# %%
# GapEncoder
# ----------
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

from skrub import GapEncoder, TableVectorizer


# %%
def plot_box_results(named_results):
    fig, ax = plt.subplots()
    names, scores = zip(
        *[(name, result["test_score"]) for name, result in named_results]
    )
    ax.boxplot(scores, vert=False)
    ax.set_yticks(range(1, len(names) + 1), labels=list(names), size=12)
    ax.set_xlabel("R2 score", size=14)
    plt.title(
        "R2 score across folds (higher is better)",
        size=14,
    )
    plt.show()


# %% Base GapEncoder
results = []

gap_pipe = make_pipeline(
    TableVectorizer(high_cardinality=GapEncoder(n_components=30)),
    HistGradientBoostingRegressor(),
)
gap_results = cross_validate(gap_pipe, X, y, scoring="r2", verbose=1)
results.append(("GapEncoder", gap_results))

# %% GapEncoder with add_words=True
gap_pipe = make_pipeline(
    TableVectorizer(high_cardinality=GapEncoder(n_components=30, add_words=True)),
    HistGradientBoostingRegressor(),
)
gap_results = cross_validate(gap_pipe, X, y, scoring="r2", verbose=1)
results.append(("GapEncoder - add_words", gap_results))

# %%
# MinHashEncoder
# --------------
from sklearn.base import clone

from skrub import MinHashEncoder

minhash_pipe = clone(gap_pipe).set_params(
    **{"tablevectorizer__high_cardinality": MinHashEncoder(n_components=30)}
)
minhash_results = cross_validate(minhash_pipe, X, y, scoring="r2", verbose=1)
results.append(("MinHashEncoder", minhash_results))

# %% TextEncoder
from skrub import TextEncoder

text_encoder = TextEncoder(
    "sentence-transformers/paraphrase-albert-small-v2",
    device="cpu",
)
text_encoder_pipe = clone(gap_pipe).set_params(
    **{"tablevectorizer__high_cardinality": text_encoder}
)
text_encoder_results = cross_validate(text_encoder_pipe, X, y, scoring="r2", verbose=1)
results.append(("TextEncoder", text_encoder_results))

# %% StringEncoder
from skrub import StringEncoder

string_encoder = StringEncoder(n_components=30, ngram_range=(3, 4), analyzer="char_wb")

string_encoder_pipe = clone(gap_pipe).set_params(
    **{"tablevectorizer__high_cardinality": string_encoder}
)
string_encoder_results = cross_validate(string_encoder_pipe, X, y, scoring="r2")
results.append(("StringEncoder - char_wb, (3,4)", string_encoder_results))

# %% Drop column
drop_pipe = clone(gap_pipe).set_params(**{"tablevectorizer__high_cardinality": "drop"})
drop_results = cross_validate(drop_pipe, X, y, scoring="r2")
results.append(("Drop", drop_results))


# %% OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value")

ordinal_encoder_pipe = clone(gap_pipe).set_params(
    **{"tablevectorizer__high_cardinality": ordinal_encoder}
)
ordinal_encoder_results = cross_validate(ordinal_encoder_pipe, X, y, scoring="r2")
results.append(("OrdinalEncoder", ordinal_encoder_results))

# %%
plot_box_results(results)


# %%
import numpy as np


def plot_performance_tradeoff(results):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    markers = ["s", "o", "^", "x", "+", "v", "1"]
    for idx, (name, result) in enumerate(results):
        ax.scatter(
            result["fit_time"],
            result["test_score"],
            label=name,
            marker=markers[idx],
        )
        mean_fit_time = np.mean(result["fit_time"])
        mean_score = np.mean(result["test_score"])
        ax.scatter(
            mean_fit_time,
            mean_score,
            color="k",
            marker=markers[idx],
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
        ax.set_ylabel("R2")
        ax.set_title("Prediction performance / training time trade-off")

    # ax.annotate(
    #     "",
    #     xy=(1.5, 0.98),
    #     xytext=(8.5, 0.90),
    #     arrowprops=dict(arrowstyle="->", mutation_scale=15),
    # )
    # ax.text(8, 0.86, "Best time / \nperformance trade-off")
    ax.legend()
    plt.show()


plot_performance_tradeoff(results)

# %%
