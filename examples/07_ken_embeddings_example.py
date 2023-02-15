"""
Machine learning with entity embeddings
=======================================

In data science, we often work with data composed of common entities,
such as cities, companies or famous people.

Embeddings, or vectorial representations of entities, are a conveniant way to
capture and summarize the information on an entity.
Recent work on relational data embeddings from Cvetkov et al.,
has provided us embeddings of all common entities from Wikipedia.
These will be called 'KEN Embeddings' in the following example. [#]_

Augmenting the data with information assembled from external
sources may be key to improving the analysis.
This is exactly the type of data we need: with embeddings of common entities,
we will be able to significantly improve our results.

.. [#] https://soda-inria.github.io/ken_embeddings/


 .. |Pipeline| replace::
     :class:`~sklearn.pipeline.Pipeline`

 .. |OneHotEncoder| replace::
     :class:`~sklearn.preprocessing.OneHotEncoder`

 .. |ColumnTransformer| replace::
     :class:`~sklearn.compose.ColumnTransformer`

 .. |MinHash| replace::
     :class:`~dirty_cat.MinHashEncoder`

 .. |HGBR| replace::
     :class:`~sklearn.ensemble.HistGradientBoostingRegressor`
"""

###############################################################################
# The data
# --------
#
# We will take a look at the Video Game sales dataset.
# We first retrieve the dataset:
import pandas as pd

X = pd.read_csv(
    "https://raw.githubusercontent.com/William2064888/vgsales.csv/main/vgsales.csv",
    sep=";",
    on_bad_lines="skip",
)
# We shuffle the data
X = X.sample(frac=1, random_state=11, ignore_index=True)
X.head(3)

###############################################################################
# Our goal will be to predict y, our target column (the sales amount):
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
# It seems better to take the log of sales rather than the sales amount itself:
import numpy as np

y = np.log(y)
sns.histplot(y)
plt.show()

###############################################################################
# Now, let's carry out some basic preprocessing:

# Get a mask of the rows with missing values in 'gender'
mask1 = X.isna()["Publisher"]
mask2 = X.isna()["Global_Sales"]
# And remove them
X.dropna(subset=["Publisher", "Global_Sales"], inplace=True)
y = y[~mask1]
y = y[~mask2]

###############################################################################
# Extracting entity embeddings
# ----------------------------
# We will use the `get_ken_embeddings` function to extract the embeddings
# of entities we need:
from dirty_cat.datasets import get_ken_embeddings

# We include all embeddings with the type name "game"
# and exclude those with type name "companies"
embedding_games = get_ken_embeddings(
    types="game", emb_id="39254360", exclude="companies|publish|develop"
)

# We include all embeddings containing the type name
# "game_development_companies", "game_companies" or "game_publish":
embedding_publisher = get_ken_embeddings(
    "game_development_companies|game_companies|game_publish",
    emb_id="39254360",
    suffix="_aux",
)

# We keep the 200 embeddings column names in a list (for the |Pipeline|):
n_dim = 200

emb_columns = []
for j in range(n_dim):
    name = "X" + str(j)
    emb_columns.append(name)

emb_columns2 = []
for j in range(n_dim):
    name = "X" + str(j) + "_aux"
    emb_columns2.append(name)

###############################################################################
# Merging the entities
# ....................
#
# We will now merge the entities from Wikipedia with their equivalent
# in our video game sales table:
#
# The entities from the 'embedding_games' table will be merged along the column "Name"
# and the ones from 'embedding_publisher' table with the column "Publisher"
from dirty_cat import FeatureAugmenter

fa1 = FeatureAugmenter(tables=[(embedding_games, "Entity")], main_key="Name")
fa2 = FeatureAugmenter(tables=[(embedding_publisher, "Entity")], main_key="Publisher")

X_full = fa1.fit_transform(X)
X_full = fa2.fit_transform(X_full)

###############################################################################
# Prediction with regular features
# --------------------------------
#
# We will forget for now the KEN Embeddings and build a typical learning
# pipeline, where will we try to predict the amount of sales only using
# the regular features contained in the initial table.

###############################################################################
# We first use scikit-learn's |ColumnTransformer| to define the columns
# that will be included in the learning process and the appropriate encoding of
# categorical variables using the |MinHash| and |OneHotEncoder|:
from sklearn.compose import make_column_transformer

from sklearn.preprocessing import OneHotEncoder
from dirty_cat import MinHashEncoder

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

all_r2_scores["Regular"] = cv_results["test_r2"]
all_rmse_scores["Regular"] = -cv_results["test_neg_root_mean_squared_error"]

print("With regular features:")
print(
    f"Mean R2 is {all_r2_scores['Regular'].mean():.2f} +-"
    f" {all_r2_scores['Regular'].std():.2f} and the RMSE is"
    f" {all_rmse_scores['Regular'].mean():.2f} +-"
    f" {all_rmse_scores['Regular'].std():.2f}"
)

###############################################################################
# Prediction with KEN Embeddings
# ------------------------------
#
# We will now build the learning |Pipeline| using only the KEN embeddings
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

all_r2_scores["KEN"] = cv_results["test_r2"]
all_rmse_scores["KEN"] = -cv_results["test_neg_root_mean_squared_error"]

print("With KEN Embeddings:")
print(
    f"Mean R2 is {all_r2_scores['KEN'].mean():.2f} +-"
    f" {all_r2_scores['KEN'].std():.2f} and the RMSE is"
    f" {all_rmse_scores['KEN'].mean():.2f} +- {all_rmse_scores['KEN'].std():.2f}"
)

###############################################################################
# Prediction with KEN Embeddings and regular features
# ---------------------------------------------------
#
# As we have seen the predictions scores in the case when embeddings are
# only present and when they are missing, we will do a final prediction
# with all variables included.

###############################################################################
# We include both the embeddings and the regular features:
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

all_r2_scores["Regular + KEN"] = cv_results["test_r2"]
all_rmse_scores["Regular + KEN"] = -cv_results["test_neg_root_mean_squared_error"]

print("With KEN Embeddings and regular features:")
print(
    f"Mean R2 is {all_r2_scores['Regular + KEN'].mean():.2f} +-"
    f" {all_r2_scores['Regular + KEN'].std():.2f} and the RMSE is"
    f" {all_rmse_scores['Regular + KEN'].mean():.2f} +-"
    f" {all_rmse_scores['Regular + KEN'].std():.2f}"
)

###############################################################################
# Plotting the results
# ....................
#
# Finally, we plot the scores on a boxplot:
plt.figure(figsize=(5, 3))
ax = sns.boxplot(data=pd.DataFrame(all_r2_scores), orient="h")
plt.ylabel("Variables", size=15)
plt.xlabel("Prediction accuracy     ", size=15)
plt.yticks(size=15)
plt.tight_layout()

###############################################################################
# There is a clear improvement when including the KEN embeddings among the
# input variables.
#
