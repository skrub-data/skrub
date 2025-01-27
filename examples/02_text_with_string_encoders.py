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

.. |HistGradientBoostingClassifier| replace::
     :class:`~sklearn.ensemble.HistGradientBoostingClassifier`

.. |RandomizedSearchCV| replace::
     :class:`~sklearn.model_selection.RandomizedSearchCV`

.. |GridSearchCV| replace::
     :class:`~sklearn.model_selection.GridSearchCV`
"""

# %%
# The Toxicity dataset
# --------------------
# We focus on the toxicity dataset, a corpus of 1,000 tweets, evenly balanced
# between the binary labels "Toxic" and "Not Toxic".
# Our goal is to classify each entry between these two labels, using only the
# text of the tweets as features.
from skrub.datasets import fetch_toxicity

dataset = fetch_toxicity()
X, y = dataset.X, dataset.y
X["is_toxic"] = y

# %%
# When it comes to displaying large chunks of text, the |TableReport| is especially
# useful! Click on any cell below to expand and read the tweet in full.
from skrub import TableReport

TableReport(X)

# %%
# GapEncoder
# ^^^^^^^^^^
# First, let's vectorize our text column using the |GapEncoder|, one of the
# `high cardinality categorical encoders <https://inria.hal.science/hal-02171256v4>`_
# provided by skrub.
# As introduced in the :ref:`previous example<example_encodings>`, the |GapEncoder|
# performs matrix factorization for topic modeling. It builds latent topics by
# capturing combinations of substrings that frequently co-occur, and encoded vectors
# correspond to topic activations.
#
# To interpret these latent topics, we select for each of them a few labels from
# the input data with the highest activations. In the example below we select 3 labels
# to summarize each topic.
from skrub import GapEncoder

gap = GapEncoder(n_components=30)
X_trans = gap.fit_transform(X["text"])
# Add the original text as a first column
X_trans.insert(0, "text", X["text"])
TableReport(X_trans)

# %%
# We can use a heatmap to highlight the highest activations, making them more visible
# for comparison against the original text and vectors above.

import numpy as np
from matplotlib import pyplot as plt


def plot_gap_feature_importance(X_trans):
    x_samples = X_trans.pop("text")

    # We slightly format the topics and labels for them to fit on the plot.
    topic_labels = [x.replace("text: ", "") for x in X_trans.columns]
    labels = x_samples.str[:50].values + "..."

    # We clip large outliers to makes activations more visible.
    X_trans = np.clip(X_trans, a_min=None, a_max=200)

    plt.figure(figsize=(10, 10), dpi=200)
    plt.imshow(X_trans.T)

    plt.yticks(
        range(len(topic_labels)),
        labels=topic_labels,
        ha="right",
        size=12,
    )
    plt.xticks(range(len(labels)), labels=labels, size=12, rotation=50, ha="right")

    plt.colorbar().set_label(label="Topic activations", size=13)
    plt.ylabel("Latent topics", size=14)
    plt.xlabel("Data entries", size=14)
    plt.tight_layout()
    plt.show()


plot_gap_feature_importance(X_trans.head())

# %%
# Now that we have an understanding of the vectors produced by the |GapEncoder|,
# let's evaluate its performance in toxicity classification. The |GapEncoder| excels
# at handling categorical columns with high cardinality, but here the column consists
# of free-form text. Sentences are generally longer, with more unique ngrams than
# high cardinality categories.
#
# To benchmark the performance of the |GapEncoder| against the toxicity dataset,
# we integrate it into a |TableVectorizer|, as introduced in the
# :ref:`previous example<example_encodings>`,
# and create a |pipeline| by appending a |HistGradientBoostingClassifier|, which
# consumes the vectors produced by the |GapEncoder|.
#
# We set ``n_components`` to 30; however, to achieve the best performance, we would
# need to find the optimal value for this hyperparameter using either |GridSearchCV|
# or |RandomizedSearchCV|. We skip this part to keep the computation time for this
# small example.
#
# Recall that the ROC AUC is a metric that quantifies the ranking power of estimators,
# where a random estimator scores 0.5, and an oracle —providing perfect predictions—
# scores 1.
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


results = []

y = X.pop("is_toxic").map({"Toxic": 1, "Not Toxic": 0})

gap_pipe = make_pipeline(
    TableVectorizer(high_cardinality=GapEncoder(n_components=30)),
    HistGradientBoostingClassifier(),
)
gap_results = cross_validate(gap_pipe, X, y, scoring="roc_auc")
results.append(("GapEncoder", gap_results))

plot_box_results(results)

# %%
# MinHashEncoder
# ^^^^^^^^^^^^^^
# We now compare these results with the |MinHashEncoder|, which is faster
# and produces vectors better suited for tree-based estimators like
# |HistGradientBoostingClassifier|. To do this, we can simply replace
# the |GapEncoder| with the |MinHashEncoder| in the previous pipeline
# using ``set_params()``.

from skrub import MinHashEncoder

minhash_pipe = make_pipeline(
    TableVectorizer(high_cardinality=MinHashEncoder(n_components=30)),
    HistGradientBoostingClassifier(),
)
minhash_results = cross_validate(minhash_pipe, X, y, scoring="roc_auc")
results.append(("MinHashEncoder", minhash_results))

plot_box_results(results)

# %%
# Remarkably, the vectors produced by the |MinHashEncoder| offer less predictive
# power than those from the |GapEncoder| on this dataset.
#
# TextEncoder
# ^^^^^^^^^^^
# Let's now shift our focus to pre-trained deep learning encoders. Our previous
# encoders are syntactic models that we trained directly on the toxicity dataset.
# To generate more powerful vector representations for free-form text and diverse
# entries, we can instead use semantic models, such as BERT, which have been trained
# on very large datasets.
#
# |TextEncoder| enables you to integrate any Sentence Transformer model from the
# Hugging Face Hub (or from your local disk) into your |pipeline| to transform a text
# column in a dataframe. By default, |TextEncoder| uses the e5-small-v2 model.
from skrub import TextEncoder

text_encoder = TextEncoder(
    "sentence-transformers/paraphrase-albert-small-v2",
    device="cpu",
)

text_encoder_pipe = make_pipeline(
    TableVectorizer(high_cardinality=text_encoder),
    HistGradientBoostingClassifier(),
)
text_encoder_results = cross_validate(text_encoder_pipe, X, y, scoring="roc_auc")
results.append(("TextEncoder", text_encoder_results))

plot_box_results(results)

# %%
# SringEncoder
# ^^^^^^^^^^^^
# |TextEncoder| embeddings are very strong, but they are also quite expensive to
# use. A simpler, faster alternative for encoding strings is the |StringEncoder|,
# which works by first performing a tf-idf (computing vectors of rescaled word
# counts of the text `wiki <https://en.wikipedia.org/wiki/Tf%E2%80%93idf>`_), and then
# following it with TruncatedSVD to reduce the number of dimensions to, in this
# case, 30.
from skrub import StringEncoder

string_encoder = StringEncoder(ngram_range=(3, 4), analyzer="char_wb")

string_encoder_pipe = make_pipeline(
    TableVectorizer(high_cardinality=string_encoder),
    HistGradientBoostingClassifier(),
)

string_encoder_results = cross_validate(string_encoder_pipe, X, y, scoring="roc_auc")
results.append(("StringEncoder", string_encoder_results))

plot_box_results(results)


# %%
# The performance of the |TextEncoder| is significantly stronger than that of
# the syntactic encoders, which is expected. But how long does it take to load
# and vectorize text on a CPU using a Sentence Transformer model? Below, we display
# the tradeoff between predictive accuracy and training time. Note that since we are
# not training the Sentence Transformer model, the "fitting time" refers to the
# time taken for vectorization.


def plot_performance_tradeoff(results):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    markers = ["s", "o", "^", "x"]
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
        ax.set_xscale("log")

        ax.set_xlabel("Time to fit (seconds)")
        ax.set_ylabel("ROC AUC")
        ax.set_title("Prediction performance / training time trade-off")

    ax.annotate(
        "",
        xy=(1.5, 0.98),
        xytext=(8.5, 0.90),
        arrowprops=dict(arrowstyle="->", mutation_scale=15),
    )
    ax.text(5.8, 0.86, "Best time / \nperformance trade-off")
    ax.legend(bbox_to_anchor=(1.02, 0.3))
    plt.show()


plot_performance_tradeoff(results)

# %%
# The black points represent the average time to fit and AUC for each vectorizer,
# and the width of the bars represents one standard deviation
#
# The green outlier dot on the right side of the plot corresponds to the first time
# the Sentence Transformers model was downloaded and loaded into memory.
# During the subsequent cross-validation iterations, the model is simply copied,
# which reduces computation time for the remaining folds.
#
# Interestingly, |StringEncoder| has a performance remarkably similar to that of
# |GapEncoder|, while being significantly faster.
#
# Conclusion
# ----------
# In conclusion, |TextEncoder| provides powerful vectorization for text, but at
# the cost of longer computation times and the need for additional dependencies,
# such as torch. |StringEncoder| represents a simpler alternative that can provide
# good performance at a fraction of the cost of more complex methods.
