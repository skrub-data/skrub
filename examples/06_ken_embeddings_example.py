"""
Wikipedia embeddings to enrich the data
=======================================

When the data comprises common entities (cities,
companies or famous people), bringing new information assembled from external
sources may be the key to improving the analysis.

Embeddings, or vectorial representations of entities, are a conveniant way to
capture and summarize the information on an entity.
Relational data embeddings capture all common entities from Wikipedia. [#]_
These will be called `KEN embeddings` in the following example.

We will see that these embeddings of common entities significantly
improve our results.

.. [#] https://soda-inria.github.io/ken_embeddings/


 .. |Pipeline| replace::
     :class:`~sklearn.pipeline.Pipeline`

 .. |OneHotEncoder| replace::
     :class:`~sklearn.preprocessing.OneHotEncoder`

 .. |ColumnTransformer| replace::
     :class:`~sklearn.compose.ColumnTransformer`

 .. |MinHash| replace::
     :class:`~skrub.MinHashEncoder`

 .. |HGBR| replace::
     :class:`~sklearn.ensemble.HistGradientBoostingRegressor`
"""

###############################################################################
# The data
# --------
#
# We will take a look at the video game sales dataset.
# Let's retrieve the dataset:
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
# Our goal will be to predict the sales amount (y, our target column):
y = X["Global_Sales"]
y

###############################################################################
# Let's take a look at the distribution of our target variable:
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="ticks")

sns.histplot(y)
plt.show()

###############################################################################
# It seems better to take the log of sales rather than the absolute values:
import numpy as np

y = np.log(y)
sns.histplot(y)
plt.show()

###############################################################################
# Before moving further, let's carry out some basic preprocessing:

# Get a mask of the rows with missing values in "Publisher" and "Global_Sales"
mask = X.isna()["Publisher"] | X.isna()["Global_Sales"]
# And remove them
X.dropna(subset=["Publisher", "Global_Sales"], inplace=True)
y = y[~mask]

###############################################################################
# Extracting entity embeddings
# ----------------------------
#
# We will use KEN embeddings to enrich our data.
#
# We will start by checking out the available tables with
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
# Next, we'll use :class:`~skrub.datasets.fetch_ken_embeddings`
# to extract the embeddings of entities we need:
from skrub.datasets import fetch_ken_embeddings

###############################################################################
# KEN Embeddings are classified by types.
# See the example on :class:`~skrub.datasets.fetch_ken_embeddings`
# to understand how you can filter types you are interested in.
#
# The :class:`~skrub.datasets.fetch_ken_embeddings` function
# allows us to specify the types to be included and/or excluded
# so as not to load all Wikipedia entity embeddings in a table.
#
#
# In a first table, we include all embeddings with the type name "game"
# and exclude those with type name "companies" or "developer".
embedding_games = fetch_ken_embeddings(
    search_types="game",
    exclude="companies|developer",
    embedding_table_id="games",
)

###############################################################################
# In a second table, we include all embeddings containing the type name
# "game_development_companies", "game_companies" or "game_publish":
embedding_publisher = fetch_ken_embeddings(
    search_types="game_development_companies|game_companies|game_publish",
    embedding_table_id="games",
    suffix="_aux",
)

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
from skrub import FeatureAugmenter

fa1 = FeatureAugmenter(tables=[(embedding_games, "Entity")], main_key="Name")
fa2 = FeatureAugmenter(tables=[(embedding_publisher, "Entity")], main_key="Publisher")

X_full = fa1.fit_transform(X)
X_full = fa2.fit_transform(X_full)

###############################################################################
# Prediction with base features
# -----------------------------
#
# We will forget for now the KEN Embeddings and build a typical learning
# pipeline, where will we try to predict the amount of sales only using
# the base features contained in the initial table.

###############################################################################
# We first use scikit-learn's |ColumnTransformer| to define the columns
# that will be included in the learning process and the appropriate encoding of
# categorical variables using the |MinHash| and |OneHotEncoder|:
from sklearn.compose import make_column_transformer

from sklearn.preprocessing import OneHotEncoder
from skrub import MinHashEncoder

min_hash = MinHashEncoder(n_components=100)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

encoder = make_column_transformer(
    ("passthrough", ["Year"]),
    (ohe, ["Genre"]),
    (min_hash, ["Platform"]),
    remainder="drop",
)

###############################################################################
# We incorporate our |ColumnTransformer| into a |Pipeline|.
# We define a predictor, |HGBR|, fast and reliable for big datasets.
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

hgb = HistGradientBoostingRegressor(random_state=0)
pipeline = make_pipeline(encoder, hgb)

###############################################################################
# The |Pipeline| can now be readily applied to the dataframe for prediction:
from sklearn.model_selection import cross_validate

# We will save the results in a dictionnary:
all_r2_scores = dict()
all_rmse_scores = dict()

cv_results = cross_validate(
    pipeline, X_full, y, scoring=["r2", "neg_root_mean_squared_error"]
)

all_r2_scores["Base features"] = cv_results["test_r2"]
all_rmse_scores["Base features"] = -cv_results["test_neg_root_mean_squared_error"]

print("With base features:")
print(
    f"Mean R2 is {all_r2_scores['Base features'].mean():.2f} +-"
    f" {all_r2_scores['Base features'].std():.2f} and the RMSE is"
    f" {all_rmse_scores['Base features'].mean():.2f} +-"
    f" {all_rmse_scores['Base features'].std():.2f}"
)

###############################################################################
# Prediction with KEN Embeddings
# ------------------------------
#
# We will now build a second learning pipeline using only the KEN embeddings
# from Wikipedia.

###############################################################################
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
# As we have seen the predictions scores in the case when embeddings are
# only present and when they are missing, we will do a final prediction
# with all variables included.

###############################################################################
# We include both the embeddings and the base features:
encoder3 = make_column_transformer(
    ("passthrough", emb_columns),
    ("passthrough", emb_columns2),
    ("passthrough", ["Year"]),
    (ohe, ["Genre"]),
    (min_hash, ["Platform"]),
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
# Plotting the results
# ....................
#
# Finally, we plot the scores on a boxplot:
plt.figure(figsize=(5, 3))
# sphinx_gallery_thumbnail_number = -1
ax = sns.boxplot(data=pd.DataFrame(all_r2_scores), orient="h")
plt.xlabel("Prediction accuracy     ", size=15)
plt.yticks(size=15)
plt.tight_layout()

###############################################################################
# There is a clear improvement when including the KEN embeddings among the
# explanatory variables.
#
# In this case, the embeddings from Wikipedia introduced
# additional background information on the game and the publisher of the
# game that would otherwise be missed.
#
# It helped significantly improve the prediction score.
#
