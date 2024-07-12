"""
.. _example_using_ken_embeddings:

==============================================
Using Wikipedia embeddings to enrich a dataset
==============================================

When the data contains common entities (cities, countries, companies,
famous people, etc.), bringing new information assembled from external sources
is a key to significantly improve the statistical analysis.

Usually, this would be done by injecting our dataset with other tables,
which we can for example download from the Internet.

In this category, Wikipedia is one of the richest sources of information when
dealing with these common entities. However, instead of downloading and joining
massive tables extracted from Wikipedia, we can instead use **embeddings**.

Embeddings, or vectorial representations of entities, are a convenient way to
capture and summarize the information related to an entity.
We will use `KEN embeddings`, which are embeddings extracted from Wikipedia
using `YAGO <https://yago-knowledge.org/>`_. [#]_

We will see how to use them in a tabular learning setting, and see whether
they improve our model's accuracy (spoiler: they do).


.. note::
    This example requires `pyarrow` to be installed.

.. [#] https://soda-inria.github.io/ken_embeddings/

.. |Pipeline| replace::
    :class:`~sklearn.pipeline.Pipeline`

.. |OneHotEncoder| replace::
    :class:`~sklearn.preprocessing.OneHotEncoder`

.. |ColumnTransformer| replace::
    :class:`~sklearn.compose.ColumnTransformer`

.. |MinHashEncoder| replace::
    :class:`~skrub.MinHashEncoder`

.. |fetch_ken_embeddings| replace::
    :class:`~skrub.datasets.fetch_ken_embeddings`

.. |HGBR| replace::
    :class:`~sklearn.ensemble.HistGradientBoostingRegressor`

.. |FeatureAugmenter| replace::
    :class:`~skrub.FeatureAugmenter`
"""

###############################################################################
# Predicting video game sales
# ---------------------------
#
# In this example, we will take a look at a video game sales dataset.
# We first retrieve our base dataset:

import pandas as pd

X = pd.read_csv(
    "https://raw.githubusercontent.com/William2064888/vgsales.csv/main/vgsales.csv",
    sep=";",
    on_bad_lines="skip",
)
# Shuffle the data
X = X.sample(frac=1, random_state=11, ignore_index=True)
X.head(3)

###############################################################################
# Our goal will be to predict our target column y, the global sales
# (in millions of copies, known via
# `cross-sourcing <https://www.gamecubicle.com/features-mario-units_sold_sales.htm>`_).

y = X["Global_Sales"]
y


###############################################################################
# Let's take a look at the distribution of our target variable:

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks")

fig, ax = plt.subplots()
sns.histplot(y, kde=True)
ax.set(xlim=(0, 10), ylim=(0, 1000))
plt.show()

###############################################################################
# It seems better to take the log of sales rather than the absolute values:

import numpy as np

y = np.log(y)

###############################################################################
# Before going further, let's clean up our dataset a bit:

# Get a mask of the rows with missing values in "Publisher" or "Global_Sales"
mask = X.isna()["Publisher"] | X.isna()["Global_Sales"]
# And remove them
X.dropna(subset=["Publisher", "Global_Sales"], inplace=True)
y = y[~mask]

###############################################################################
# Extracting entity embeddings
# ----------------------------
#
# We will use a subset of the KEN embeddings -- specific to the video game
# industry -- that was extracted previously.
#
# We will start by checking out the available subtables with
# :class:`~skrub.datasets.fetch_ken_table_aliases`:

from skrub.datasets import fetch_ken_table_aliases

fetch_ken_table_aliases()

###############################################################################
# The *games* table is the most relevant to our case.
# Let's see what kind of types we can find in it with the function
# :class:`~skrub.datasets.fetch_ken_types`:

from skrub.datasets import fetch_ken_types

fetch_ken_types(embedding_table_id="games")

###############################################################################
# Interesting, we have a broad range of topics!
#
# We will use the |fetch_ken_embeddings| function to extract the embeddings
# of entities we need:

from skrub.datasets import fetch_ken_embeddings

###############################################################################
# KEN Embeddings are classified by types.
# The `fetch_ken_embeddings` function allows us to specify the types to be
# included and/or excluded so as not to load all Wikipedia entity embeddings in a table.
#
#
# In a first table, we include all embeddings with the type name "game"
# and exclude those with type name "companies" or "developer".
embedding_games = fetch_ken_embeddings(
    search_types="game",
    exclude="companies|developer",
    embedding_table_id="games",
)

embedding_games.head()

###############################################################################
# In a second table, we include all embeddings containing the type name
# "game_development_companies", "game_companies" or "game_publish":
embedding_publisher = fetch_ken_embeddings(
    search_types="game_development_companies|game_companies|game_publish",
    embedding_table_id="games",
)

embedding_publisher.head()

###############################################################################
# We keep the 200 embeddings column names in a list (for the |Pipeline|):
n_dim = 200

emb_columns = [f"X{j}" for j in range(n_dim)]

emb_columns2 = [f"X{j}_aux" for j in range(n_dim)]

###############################################################################
# Merging the entities
# ....................
#
# We will now merge the entities from Wikipedia with their equivalent match
# in our video game sales table:
#
# The entities from the 'embedding_games' table will be merged along the
# column "Name" and the ones from 'embedding_publisher' table with the
# column "Publisher"
from skrub import Joiner

fa1 = Joiner(embedding_games, aux_key="Entity", main_key="Name")
fa2 = Joiner(embedding_publisher, aux_key="Entity", main_key="Publisher", suffix="_aux")

X_full = fa1.fit_transform(X)
X_full = fa2.fit_transform(X_full)

###############################################################################
# Prediction with base features
# -----------------------------
#
# We will put the KEN Embeddings aside for now, and build a machine learning
# pipeline, where will we try to predict the global sales using only
# the base features contained in our initial table.
#
# We first use scikit-learn's |ColumnTransformer| to define the columns
# that will be included in the learning process and the appropriate encoding of
# categorical variables using the |MinHashEncoder| and |OneHotEncoder|:

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

from skrub import MinHashEncoder

min_hash = MinHashEncoder(n_components=100)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

encoder = make_column_transformer(
    ("passthrough", ["Year"]),
    (ohe, ["Genre"]),
    (min_hash, "Platform"),
    remainder="drop",
)

###############################################################################
# We incorporate our |ColumnTransformer| into a |Pipeline|.
# We define a predictor -- |HGBR| -- fast and reliable for big datasets.

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

hgb = HistGradientBoostingRegressor(random_state=0)
pipeline = make_pipeline(encoder, hgb)

###############################################################################
# The |Pipeline| can now be applied to the dataframe for prediction:

from sklearn.model_selection import cross_validate

# We will save the results in dictionaries:
all_r2_scores = dict()
all_rmse_scores = dict()

cv_results = cross_validate(
    pipeline, X_full, y, scoring=["r2", "neg_root_mean_squared_error"]
)

all_r2_scores["Base features"] = cv_results["test_r2"]
all_rmse_scores["Base features"] = -cv_results["test_neg_root_mean_squared_error"]

print("With base features:")
print(
    f"Mean R2 is {all_r2_scores['Base features'].mean():.2f} +- "
    f"{all_r2_scores['Base features'].std():.2f} and the RMSE is "
    f"{all_rmse_scores['Base features'].mean():.2f} +- "
    f"{all_rmse_scores['Base features'].std():.2f}. "
)

###############################################################################
# Prediction with KEN Embeddings
# ------------------------------
#
# We will now build a second learning pipeline using only the KEN embeddings
# from Wikipedia.
#
# We keep only the embeddings columns:

encoder2 = make_column_transformer(
    ("passthrough", emb_columns), ("passthrough", emb_columns2), remainder="drop"
)

###############################################################################
# We redefine the |Pipeline|:

pipeline2 = make_pipeline(encoder2, hgb)

###############################################################################
# Let's look at the results:

cv_results = cross_validate(
    pipeline2, X_full, y, scoring=["r2", "neg_root_mean_squared_error"]
)

all_r2_scores["KEN features"] = cv_results["test_r2"]
all_rmse_scores["KEN features"] = -cv_results["test_neg_root_mean_squared_error"]

print("With KEN Embeddings:")
print(
    f"Mean R2 is {all_r2_scores['KEN features'].mean():.2f} +-"
    f" {all_r2_scores['KEN features'].std():.2f} and the RMSE is"
    f" {all_rmse_scores['KEN features'].mean():.2f} +-"
    f" {all_rmse_scores['KEN features'].std():.2f}"
)

###############################################################################
# It seems including the embeddings is very relevant for the prediction task
# at hand!

###############################################################################
# Prediction with KEN Embeddings and base features
# ------------------------------------------------
#
# As we have seen the predictions scores both when embeddings used exclusively,
# and when they are missing entirely.
# We will now do a final prediction with all the features we have:

encoder3 = make_column_transformer(
    ("passthrough", emb_columns),
    ("passthrough", emb_columns2),
    ("passthrough", ["Year"]),
    (ohe, ["Genre"]),
    (min_hash, "Platform"),
    remainder="drop",
)

###############################################################################
# We redefine the |Pipeline|:

pipeline3 = make_pipeline(encoder3, hgb)

###############################################################################
# Let's look at the results:

cv_results = cross_validate(
    pipeline3, X_full, y, scoring=["r2", "neg_root_mean_squared_error"]
)

all_r2_scores["Base + KEN features"] = cv_results["test_r2"]
all_rmse_scores["Base + KEN features"] = -cv_results["test_neg_root_mean_squared_error"]

print("With KEN Embeddings and base features:")
print(
    f"Mean R2 is {all_r2_scores['Base + KEN features'].mean():.2f} +-"
    f" {all_r2_scores['Base + KEN features'].std():.2f} and the RMSE is"
    f" {all_rmse_scores['Base + KEN features'].mean():.2f} +-"
    f" {all_rmse_scores['Base + KEN features'].std():.2f}"
)

###############################################################################
# Comparing the difference
# ........................
#
# Finally, we plot the scores on a boxplot:

plt.figure(figsize=(5, 3))
ax = sns.boxplot(data=pd.DataFrame(all_r2_scores), orient="h")
plt.xlabel("Prediction accuracy     ", size=15)
plt.yticks(size=15)
plt.tight_layout()

###############################################################################
# There is a clear improvement when including the KEN embeddings among the
# explanatory variables.
#
# Conclusion
# ----------
#
# In this case, the embeddings from Wikipedia introduced
# additional background information on the game and the publisher of the
# game that would otherwise be missed, which helped significantly
# for improving the prediction score.
