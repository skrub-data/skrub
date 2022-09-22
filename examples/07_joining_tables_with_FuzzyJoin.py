"""
Merging a collection of dirty tables
====================================

When combining data from different sources, there is a risk that
it will not be easily merged, as it comes mislabeled, with errors, duplicated.

In this example, we show how the :func:`fuzzy_join` function allows us to join
tables without cleaning the data and but by taking into account the
label variations.

Simple and time-saving, this method is intended for users to apply
before training their machine learning model.

To illustrate, we will join data from the `2022 World Happiness Report <https://worldhappiness.report/>`_.
with tables provided in `the World Bank open data platform <https://data.worldbank.org/>`_
in order to create a satisfying prediction model.

"""

###############################################################################
# Data Importing and preprocessing
# --------------------------------
#
# We import the happiness score data first:
import pandas as pd

df = pd.read_csv(
    "https://raw.githubusercontent.com/dirty-cat/datasets/master/data/Happiness_report_2022.csv",
    thousands=",",
)
df.drop(df.tail(1).index, inplace=True)

# Let's take a look at the table:
df.head(3)

##############################################################################
# The Happiness score was computed using the Gallup World Poll survey results.
# The report stress out some of the possible explanatory factors: GDP per
# capita, Social support, Generosity etc.
###############################################################################
# If we want to create a machine learning model which predicts
# the happiness index of any new country or future date,
# we will need to include explanatory factors from other tables.
# What can we add from the available online public data tables?

###############################################################################
# Finding additional tables
# -------------------------
#
# Let's inspire ourselfes from the factors used by the Happiness report to
# find additional features.
# We will extract data from the World Bank (WB) databank.
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

###############################################################################
# Joining World Bank tables to our initial one
# ----------------------------------------------
#
# So now we have our initial table and 3 additional ones that we have
# extracted.
#
# To join them with dirty_cat, we only need to do the following:
from dirty_cat import fuzzy_join

X1 = fuzzy_join(X, gdppc, left_on="Country", right_on="Country Name", return_score=True)
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

###############################################################################
# Keeping only the good matches
# ------------------------------
#################################################################
# The best way to inspect the matches is to use the following function:
import numpy as np


def print_worst_matches(joined_table, n=5):
    """Prints n worst matches for inspection."""
    max_ind = np.argpartition(joined_table["distance"], n, axis=0)[:n]
    max_dist = pd.Series(
        joined_table["distance"][max_ind.ravel()].ravel(), index=max_ind.ravel()
    )
    worst_matches = joined_table.iloc[list(max_ind.ravel())]
    worst_matches = worst_matches.assign(distance=max_dist)
    print("The worst five matches are the following:\n")
    return worst_matches


# This will print out the five worst matches, which will give
# us an overview of the situation:


print_worst_matches(X1, n=4)
# We see that some matches were unsuccesful
# (e.g 'Palestinian Territories*' and 'Estonia'),
# because there is simply no match in the two tables.

#################################################################
#
# In this case, it is better to use the threshold parameter
# so as to include only precise-enough matches:
#
# TODO: improve threshold measurement, here it excludes some good matches as well:
X1 = fuzzy_join(
    X,
    gdppc,
    left_on="Country",
    right_on="Country Name",
    match_score=0.45,
    return_score=True,
)
print_worst_matches(X1, n=4)
# Matches that are not available (or precise enough) are marked as `NaN`.
X1.drop(["distance"], axis=1, inplace=True)

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
# We now only remove categories with missing information:
mask = X3["GDP per capita (current US$)"].notna()
y = np.ravel(y[mask])

X1 = X1[mask]
X2 = X2[mask]
X3 = X3[mask]

#################################################################
# And we are ready to apply a machine learning model to it!

###############################################################################
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
cv = KFold(n_splits=4, shuffle=True, random_state=0)

#################################################################
# To evaluate our model, we will apply a `4-fold cross-validation`.
# We evaluate our model using the `R2` score.
#
# Let's finally assess the results of our models:
from sklearn.model_selection import cross_validate

for data in (X1, X2, X3):
    cv_results_t = cross_validate(
        hgdb, data.select_dtypes(exclude=object), y, cv=cv, scoring="r2"
    )
    cv_r2_t = cv_results_t["test_score"]
    print(
        f"Mean R2 score with {len(data.columns) - 2} feature columns is"
        f" {cv_r2_t.mean():.2f} +- {cv_r2_t.std():.2f}"
    )

#################################################################
# Our score gets better every time we add additional information into our
# table!
#
# Data cleaning varies from dataset to dataset: there are as
# many ways to clean a table as there are errors. :func:`fuzzy_join`
# method is generalizable across all datasets.
#
# Data transformation is also often very costly in both time and ressources.
# :func:`fuzzy_join` is fast and easy-to-use.
#
