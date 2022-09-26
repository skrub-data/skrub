"""
Merging a collection of dirty tables
====================================

When combining data from different sources, there is a risk that
it will not be easily merged, as it comes mislabeled, with errors, duplicated.

In this example, we show how the :func:`fuzzy_join` function allows us to join
tables without cleaning the data by taking into account the
label variations.

Simple and time-saving, this method is intended for users to apply
before training their machine learning model.

To illustrate, we will join data from the `2022 World Happiness Report <https://worldhappiness.report/>`_.
with tables provided in `the World Bank open data platform <https://data.worldbank.org/>`_
in order to create a satisfying first prediction model.

"""

#######################################################################
# Data Importing and preprocessing
# --------------------------------
#
# We import the happiness score table first:
import pandas as pd

df = pd.read_csv(
    "https://raw.githubusercontent.com/dirty-cat/datasets/master/data/Happiness_report_2022.csv",
    thousands=",",
)
df.drop(df.tail(1).index, inplace=True)

# Let's take a look at the table:
df.head(3)

##############################################################################
# The is a table that contains the happiness index of a country along with
# some of the possible explanatory factors: GDP per capita, Social support,
# Generosity etc.
# For more information, read the `World Happiness Report website <https://worldhappiness.report/>`_.
X = df[["Country"]]
y = df[["Happiness score"]]
# We keep the country names in our X table and we create
# the y table with the happiness score (our prediction target).

############################################################################
# Now, we will need to include explanatory factors from other tables.
# What can we add from the available online public data tables to complete
# our X table?

#############################################################################
# Finding additional tables
# -------------------------
#
# Interesting tables can be found on the the World Bank (WB) databank.
# We will extract data from their website and include them in our model.
# Luckily, dirty_cat has the following function to do it easily:
from dirty_cat.datasets import fetch_world_bank_indicator

#################################################################
# We extract the table containing GDP per capita by country:
gdppc = fetch_world_bank_indicator(indicator_id="NY.GDP.PCAP.CD").X
gdppc.head(3)

#################################################################
# Then another table, with life expectancy by country:
life_exp = fetch_world_bank_indicator("SP.DYN.LE00.IN", "life_exp").X
life_exp.head(3)

#################################################################
# And a table with legal rights strength by country:
legal_rights = fetch_world_bank_indicator("IC.LGL.CRED.XQ").X
legal_rights.head(3)

#######################################################################
# Joining World Bank tables to our initial one
# ----------------------------------------------
#
# So now we have our initial table, X, and 3 additional ones that we have
# extracted.
#
# To join them with dirty_cat, we only need to do the following:
from dirty_cat import fuzzy_join

X1 = fuzzy_join(
    X,  # our table to join
    gdppc,  # the table to join with
    left_on="Country",  # the first join key column
    right_on="Country Name",  # the second join key column
    return_score=True,
)

X1.head(20)
# We merged the first WB table to our initial one.

#################################################################
# .. topic:: Note:
#
#    We fix the `return_score` parameter to `True` so as to keep the matching
#    score, that we will use later to establish the worst matches.

#################################################################
#
# We see that our :func:`fuzzy_join` succesfully identified the countries,
# even though some country names differ between tables.
#
# For instance, 'Czechia' is well identified as 'Czech Republic' and
# 'Luxembourg*' as 'Luxembourg'.
#
# .. topic:: Note:
#
#    This would all be missed out if we were using other methods such as
#    `pandas.DataFrame.join <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html>`_,
#    which searches only for exact matches.
#    In this case, to reach the best result, we would have to manually clean
#    the data (e.g. remove the * after country name) and look manually
#    for matching patterns in observations.
#
# Let's do some more inspection of the merging done.

######################################################################
# Keeping only the good matches
# ------------------------------
#################################################################
# The best way to inspect the matches is to use the following function:
import numpy as np


def print_worst_matches(joined_table, n=5):
    """Prints n worst matches for inspection."""
    max_ind = np.argpartition(joined_table["matching_score"], n, axis=0)[:n]
    max_dist = pd.Series(
        joined_table["matching_score"][max_ind.ravel()].ravel(), index=max_ind.ravel()
    )
    worst_matches = joined_table.iloc[list(max_ind.ravel())]
    worst_matches = worst_matches.assign(distance=max_dist)
    print("The worst five matches are the following:\n")
    return worst_matches


# This will print out the five worst matches, which will give
# us an overview of the situation:


print_worst_matches(X1, n=4)
# We see that some matches were unsuccesful
# (e.g 'Palestinian Territories*' and 'Palau'),
# because there is simply no match in the two tables.

#################################################################
#
# In this case, it is better to use the threshold parameter
# so as to include only precise-enough matches:
#
X1 = fuzzy_join(
    X,
    gdppc,
    left_on="Country",
    right_on="Country Name",
    match_score=0.35,
    return_score=True,
)
print_worst_matches(X1, n=4)
# Matches that are not available (or precise enough) are marked as `NaN`.

#################################################################
#
# Now let's include other information that may be relevant, such as
# life expectancy table:
X2 = fuzzy_join(
    X1,
    life_exp,
    left_on="Country",
    right_on="Country Name",
    match_score=0.45,
    how="left",
)

X2.head(3)
#################################################################
# .. topic:: Note:
#
#    Here, we use the `keep='left'` option to keep only the left key matching
#    column, so as not to have too much unnecessary columns with country names.
#
# And the table with a measure of legal rights strenght in the country:
X3 = fuzzy_join(
    X2,
    legal_rights,
    left_on="Country",
    right_on="Country Name",
    match_score=0.45,
    how="left",
)

X3.head(3)
#################################################################
#
# Great! Our joined table has became bigger and full of useful informations.
# We now only remove missing or unused information:
X3.drop(["matching_score"], axis=1, inplace=True)
mask = X3["GDP per capita (current US$)"].notna()
X3 = X3[mask]

y = np.ravel(y[mask])

#################################################################
# And we are ready to apply a first machine learning model to it!

###################################################################
# Prediction model
# -----------------
#
# Let us now define the model that will be used to predict the happiness score:
from sklearn import __version__ as sklearn_version
from dirty_cat._utils import Version

if Version(sklearn_version) < Version("1.0"):
    from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

hgdb = HistGradientBoostingRegressor(random_state=0)
cv = KFold(n_splits=2, shuffle=True, random_state=0)

#################################################################
# To evaluate our model, we will apply a `4-fold cross-validation`.
# We evaluate our model using the `R2` score.
#
# Let's finally assess the results of our models:
from sklearn.model_selection import cross_validate

cv_results_t = cross_validate(
    hgdb, X3.select_dtypes(exclude=object), y, cv=cv, scoring="r2"
)
cv_r2_t = cv_results_t["test_score"]
print(
    f"Mean R2 score with {len(X3.columns) - 2} feature columns is"
    f" {cv_r2_t.mean():.2f} +- {cv_r2_t.std():.2f}"
)

#################################################################
# We have a satisfying first result: an R2 of 0.63!
#
# Data cleaning varies from dataset to dataset: there are as
# many ways to clean a table as there are errors. :func:`fuzzy_join`
# method is generalizable across all datasets.
#
# Data transformation is also often very costly in both time and ressources.
# :func:`fuzzy_join` is fast and easy-to-use.
#
# Now up to you, try improving our model by adding information into it and
# beat our result!
