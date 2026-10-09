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

.. |LLMEncoder| replace::
     :class:`~skrub.LLMEncoder`

.. |StringEncoder| replace::
     :class:`~skrub.StringEncoder`

.. |CatEncoder| replace::
     :class:`~skrub.CatEncoder`

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
import pandas as pd

from skrub.datasets import fetch_toxicity

# %%
# We load the dataset from the path using pandas.
file_path = fetch_toxicity().path

X = pd.read_csv(file_path)

# %%
# When it comes to displaying large chunks of text, the |TableReport| is especially
# useful! Click on any cell below to expand and read the tweet in full.
from skrub import TableReport

TableReport(X)

# %%
# We prepare the target variable by mapping the binary labels "Toxic" and "Not Toxic"
# to 1 and 0, respectively. The target is reused throughout the example.

y = X.pop("is_toxic").map({"Toxic": 1, "Not Toxic": 0})

# %%
#
# To benchmark the performance of the various encoders against the toxicity dataset,
# we integrate them into a |TableVectorizer|, as introduced in the
# :ref:`previous example<example_encodings>`,
# and create a |pipeline| by appending a |HistGradientBoostingClassifier|, which
# consumes the vectors produced by each encoder.
#
# We set ``n_components`` of each encoder to 30; however, to achieve the best
# performance, we would need to find the optimal value for this hyperparameter
# using either |GridSearchCV| or |RandomizedSearchCV|. We skip this part to keep
# the computation time for this small example.
#
# Recall that the ROC AUC is a metric that quantifies the ranking power of estimators,
# where a random estimator scores 0.5, and an oracle —providing perfect predictions—
# scores 1.
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

from skrub import TableVectorizer

# %%
# We use a boxplot to visualize the distribution of ROC AUC scores across folds
# for each encoder.
import matplotlib.pyplot as plt
def plot_box_results(named_results):
    fig, ax = plt.subplots()
    names, scores = zip(
        *[(name, result["test_score"]) for name, result in named_results]
    )
    ax.boxplot(scores, orientation="horizontal")
    ax.set_yticks(range(1, len(names) + 1), labels=list(names), size=12)
    ax.set_xlabel("ROC AUC", size=14)
    ax.set_title(
        "AUC distribution across folds (higher is better)",
        size=14,
    )
    plt.show()

results = []

# %%
# StringEncoder
# ^^^^^^^^^^^^^
# First, let's vectorize our text column using the |StringEncoder|, which is a
# simple and fast encoder for strings, and is used as the default encoder for
# string columns in skrub.
#
# The |StringEncoder| works by first performing a tf-idf
# (computing vectors of rescaled word counts of the text
# `wiki <https://en.wikipedia.org/wiki/Tf%E2%80%93idf>`_), and then
# following it with TruncatedSVD to reduce the number of dimensions to, in this
# case, 30.
# The |StringEncoder| can typically produce good quality vectors for text and is
# quite fast to compute.

from skrub import StringEncoder

string_encoder = StringEncoder(ngram_range=(3, 4), analyzer="char_wb", random_state=0)

string_encoder_pipe = make_pipeline(
    TableVectorizer(high_cardinality=string_encoder),
    HistGradientBoostingClassifier(),
)

string_encoder_results = cross_validate(string_encoder_pipe, X, y, scoring="roc_auc")
results.append(("StringEncoder", string_encoder_results))

plot_box_results(results)

# %%
# LLMEncoder
# ^^^^^^^^^^^
# A far more powerful alternative to the |StringEncoder| is the |LLMEncoder|, which
# leverages pre-trained deep learning models to generate vector representations of text.
# The |StringEncoder| and |CatEncoder| are syntactic models that we trained directly
# on the toxicity dataset.
# The |LLMEncoder| is a semantic model that has been trained on a large corpus of
# text, allowing it to capture the meaning and context of words and phrases.
# To generate more powerful vector representations for free-form text and diverse
# entries, we can instead use semantic models, such as BERT, which have been trained
# on very large datasets.
#
# |LLMEncoder| enables you to integrate any Sentence Transformer model from the
# Hugging Face Hub (or from your local disk) into your |pipeline| to transform a text
# column in a dataframe. By default, |LLMEncoder| uses the e5-small-v2 model.
from skrub import LLMEncoder

llm_encoder = LLMEncoder(
    "sentence-transformers/paraphrase-albert-small-v2",
    device="cpu",
)

llm_encoder_pipe = make_pipeline(
    TableVectorizer(high_cardinality=llm_encoder),
    HistGradientBoostingClassifier(),
)
llm_encoder_results = cross_validate(llm_encoder_pipe, X, y, scoring="roc_auc")
results.append(("LLMEncoder", llm_encoder_results))

plot_box_results(results)

# %%
# GapEncoder
# ^^^^^^^^^^
# We now evaluate the performance of the |GapEncoder|
# (`reference paper <https://inria.hal.science/hal-02171256v4>`_),
# a high cardinality encoder that performs matrix factorization for topic modeling.
# The |GapEncoder| builds latent topics by capturing combinations of substrings
# that frequently co-occur, and encoded vectors correspond to topic activations.
# The |GapEncoder| typically works well for categorical columns with high
# cardinality, but here the column consists of free-form text.
# Sentences are generally longer, with more unique ngrams than high cardinality
# categories.
from skrub import GapEncoder

gap = GapEncoder(n_components=30)
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
# The |MinHashEncoder| is faster and produces vectors better suited for
# tree-based estimators like |HistGradientBoostingClassifier|.

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
# power than those from all the other encoders, despite being faster to compute.
#

# %%
# CatEncoder
# ^^^^^^^^^^^^^^
# The |CatEncoder| is a high cardinality encoder that uses a combination of
# |OneHotEncoder| and |TargetEncoder| to produce vectors for categorical columns.
# Specifically, |TargetEncoder| is added to the |OneHotEncoder| to produce a
# vector representation of each category based on the target variable; rare
# categories are marked as "infrequent".

from skrub import CatEncoder

cat_pipe = make_pipeline(
    TableVectorizer(high_cardinality=CatEncoder()),
    HistGradientBoostingClassifier(),
)
cat_results = cross_validate(cat_pipe, X, y, scoring="roc_auc")
results.append(("CatEncoder", cat_results))

plot_box_results(results)

# %%
# In this case, the |CatEncoder| cannot learn anything from the dataset: since
# all the entries are unique, the |CatEncoder| cannot find any patterns in the data,
# and for this reason its encodings are not useful for the classification task.


# %%
# Performance tradeoff
# ------------------------
# The performance of the |LLMEncoder| is significantly stronger than that of
# the syntactic encoders, which is expected. But how long does it take to load
# and vectorize text on a CPU using a Sentence Transformer model? Below, we display
# the tradeoff between predictive accuracy and training time. Note that since we are
# not training the Sentence Transformer model, the "fitting time" refers to the
# time taken for vectorization.

import numpy as np

def plot_performance_tradeoff(results):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    markers = ["s", "o", "^", "x", "D"]
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
        "Best time / \nperformance trade-off",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        xytext=(0.2, 0.8),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.5, mutation_scale=15),
    )
    ax.legend(bbox_to_anchor=(1.02, 0.3))
    plt.show()


plot_performance_tradeoff(results)

# %%
# The black points represent the average time to fit and AUC for each vectorizer,
# and the width of the bars represents one standard deviation.
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
# In conclusion, |LLMEncoder| provides powerful vectorization for text, but at
# the cost of longer computation times and the need for additional dependencies,
# such as torch. |StringEncoder| represents a simpler alternative that can provide
# good performance at a fraction of the cost of more complex methods.
