"""
Joining tables with fuzzy_join
================================

In this example, we show how to join tables with the :func:`fuzzy_join` function.
We also demonstrate why this method is the most easy and appropriate tool for handling
the joining of tables for users that want to improve their machine learning models quickly.

We will illustrate the join to predict the happiness score of a country from
the `2022 World Happiness Report <https://worldhappiness.report/>`_.
We will also use data provided from `the World Bank open data platform <https://data.worldbank.org/>`_
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

#################################################################
# The Happiness score was computed using the Gallup World Poll survey results.
# The report stress out some of the possible explanatory factors: GDP per capita, Social support, Generosity etc.
# However, these factors here are only estimated indexes used to calculate the happiness score.
# Thus, we will not use them for our prediction model.
###############################################################################
# The sum of all explanatory indexes is then the happiness score itself:
df["Sum_of_factors"] = df.iloc[:, [5, 6, 7, 8, 9, 10, 11]].sum(axis=1)
df[["Happiness score", "Sum_of_factors"]].head(3)
#################################################################
X = df[["Country"]]
y = df[["Happiness score"]]
# We defined our X and y variables.
###############################################################################
# If we want to create a machine learning model which predicts
# the happiness index of any new country or future date,
# we will need to include explanatory factors from other tables.

###############################################################################
# Finding additional tables
# ---------------------------
#
# Let's inspire ourselfes from the factors used by the Happiness report to explain happiness.
# We will extract data from the World Bank databank using the following function:
from dirty_cat.datasets import fetch_world_bank_indicator

#################################################################
# We then extract GDP per capita by country:
gdppc = fetch_world_bank_indicator(indicator_id="NY.GDP.PCAP.CD").X
gdppc.head(3)

#################################################################
# Life expectancy by country:
life_exp = fetch_world_bank_indicator("SP.DYN.LE00.IN", "life_exp").X
life_exp.head(3)

#################################################################
# And the legal rights strength by country:
legal_rights = fetch_world_bank_indicator("IC.LGL.CRED.XQ").X
legal_rights.head(3)

###############################################################################
# Joining World Bank tables to our initial one
# ----------------------------------------------
#
# Now, using dirty_cat's :func:`fuzzy_join` function,
# we need only one line to join two tables
# without worrying about preprocessing:
#
# We add GDP per capita to the initial table:
from dirty_cat import fuzzy_join

X1 = fuzzy_join(X, gdppc, left_on="Country", right_on="Country Name", return_score=True)
X1.head(20)

#################################################################
# .. topic:: Note:
#
#    We fix the `return_score` parameter to `True` so as to keep the matching
#    score, that we will use later to establish the worst matches.

#################################################################
#
# Now, we see that our :func:`fuzzy_join` succesfully identified the countries,
# even though some country names differ between tables.
#
# For instance, 'Czechia' is well identified as 'Czech Republic' and 'Luxembourg*' as 'Luxembourg'.
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
# Dirty_cat's :func:`fuzzy_join` is the perfect function to avoid doing so (and save time) with great results.

###############################################################################
# Keeping only the good matches
# ------------------------------
#################################################################
# The best way to inspect the matches is to use the `print_worst_matches` function.
# This will print out the five worst matches, which will give us an overview of the situation:
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


print_worst_matches(X1, n=4)
# We see that some matches were unsuccesful (e.g 'Palestinian Territories*' and 'Estonia'),
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
# Matches that are not available (or precise enough) are thus marked as `NaN`.
X1.drop(["distance"], axis=1, inplace=True)
#################################################################
#
# Now let's include other information that may be relevant, such as life expectancy:
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
#    Here, we use the `keep='left'` option to keep only the left key matching column,
#    so as not to have too much unnecessary columns with country names.
#
# And the strenght of legal rights in the country:
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
# Great! Our table has became bigger and full of useful informations.
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
# ---------------------
#
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
# Our score gets better every time we add additional information into our table!
#
# This is why dirty_cat's :func:`fuzzy_join` is an easy-to-use
# and useful tool.
