"""
.. _example_fuzzy_joining:

==========================
Fuzzy joining dirty tables
==========================

In this notebook, we demonstrate how to combine non-normalized tables
from different sources.

Joining this data is difficult: entries on one side do not necessarily
have exact matches on the other side.

The |fuzzy_join| function enables us to join tables without cleaning the data,
by accounting for the vocabulary variations.

To illustrate that, we will join data from the
`2022 World Happiness Report <https://worldhappiness.report/>`_
with tables provided by the
`World Bank open data platform <https://data.worldbank.org/>`_
in order to create a prediction model.


.. |fuzzy_join| replace::
    :func:`~skrub.fuzzy_join`

.. |FeatureAugmenter| replace::
    :class:`~skrub.FeatureAugmenter`
"""

###############################################################################
# Importing the data
# ------------------
#
# Let's import both our tables ; first, the happiness score dataset:

import pandas as pd

df = pd.read_csv(
    "https://raw.githubusercontent.com/dirty-cat/datasets/master/data/Happiness_report_2022.csv",
    thousands=",",
)
# It contains a placeholder row at the end, that we remove:
df = df.drop(df.tail(1).index)

###############################################################################
# This table contains the happiness index of countries along with
# some of the possible explanatory factors: GDP per capita, Social support,
# Generosity etc.

df.head(5)

###############################################################################
# .. topic:: Note:
#
#     For the sake of the example, we will ignore the other variables.
#     We will go as far as remove them, so they don't impact the accuracy
#     of the model we'll train later on.
#
# We now want to enrich our main table with additional explanatory factors.
#
# Interesting tables can be found on `the World Bank open data platform
# <https://data.worldbank.org/>`_, for which skrub has a download function:

from skrub.datasets import fetch_world_bank_indicator

###############################################################################
# Let's fetch a table containing the life expectancy by country:

life_expectancy = fetch_world_bank_indicator("SP.DYN.LE00.IN").X
life_expectancy.head(5)

###############################################################################
# We extract the table containing GDP per capita by country:

gdp_per_capita = fetch_world_bank_indicator(indicator_id="NY.GDP.PCAP.CD").X
gdp_per_capita.head(5)

###############################################################################
# And a table with legal rights strength by country:

legal_rights = fetch_world_bank_indicator("IC.LGL.CRED.XQ").X
life_expectancy.head(5)

###############################################################################
# A correspondance problem
# ------------------------
#
# Alas, the entries for countries do not perfectly match between our
# original table, and those that we downloaded from the WorldBank.

df.sort_values(by="Country").tail(7)

###############################################################################

life_expectancy.sort_values(by="Country Name").tail(7)

###############################################################################
# For example, we can see Yemen written "Yemen*" on one side, and
# "Yemen, Rep." on the other.
#
# We also have entries that probably don't have correspondances: "World"
# on one side, whereas the other table only has country-level data.

###############################################################################
# Joining tables with imperfect correspondance
# --------------------------------------------
#
# We will now join one of our three tables, which, despite the mismatches,
# can be easily done with skrub:

from skrub import fuzzy_join

df1 = fuzzy_join(
    left=df,
    right=life_expectancy,
    left_on="Country",  # the first table join key column
    right_on="Country Name",  # the second table join key column
    return_score=True,
)
df1.tail(20)

###############################################################################
# .. topic:: Note:
#
#     We fix the ``return_score`` parameter to `True` to keep the matching
#     score, which we'll use to investigate what are the worst matches.
#
# We see that |fuzzy_join| successfully matched the countries,
# despite some country names differing between tables.
#
# For instance, "Czechia" is well identified as "Czech Republic" and
# "Luxembourg*" as "Luxembourg".
#
# .. topic:: Note:
#
#     These correspondances would be missed by traditional methods such as
#     `pandas.merge <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html>`_,
#     as it can only join on exact matches.
#     Using this tool, in order to reach the best result, we would have to
#     `manually` clean the data (e.g. remove the * after country name)
#     and look for matching patterns in every observation.
#
# Let's inspect the merge result: we will print the worst matches

df1.sort_values("matching_score").head(5)

###############################################################################
# We see that some matches are incorrect
# (e.g "Palestinian Territories*" and "Palau"),
# because there is simply no match in the two tables.
#
# Here, it would better to use the threshold parameter to exclude
# too imprecise matches:

df1 = fuzzy_join(
    left=df,
    right=life_expectancy,
    left_on="Country",
    right_on="Country Name",
    match_score=0.65,
    return_score=True,
)
df1.sort_values("matching_score").head(5)

###############################################################################
# Matches that are not available (or precise enough) are marked as `NaN`.
# We can automatically remove by passing 'drop_unmatched':

df1 = fuzzy_join(
    left=df,
    right=life_expectancy,
    left_on="Country",
    right_on="Country Name",
    match_score=0.65,
    drop_unmatched=True,
)
df1.drop(columns=["Country Name"], inplace=True)

###############################################################################
# We can now plot and look at the link between life expectancy
# and happiness:

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("notebook")

plt.figure(figsize=(4, 3))
ax = sns.regplot(
    data=df1,
    x="Life expectancy at birth, total (years)",
    y="Happiness score",
    lowess=True,
)
ax.set_ylabel("Happiness index")
ax.set_title("Is a higher life expectancy linked to happiness?")
plt.tight_layout()
plt.show(5)

###############################################################################
# It seems that happiness is correlated to life expectancy.
# However, it is certainly not the only explanatory factor.
# We have to search for other patterns.
#
# Let's include the other tables we got earlier.
# First, the GPT per capita information:

df2 = fuzzy_join(
    left=df1,
    right=legal_rights,
    left_on="Country",
    right_on="Country Name",
    match_score=0.65,
)
df2.head(5)

###############################################################################
# Let's take a look at the correspondance in a figure:

plt.figure(figsize=(4, 3))
ax = sns.regplot(
    data=df2,
    x="GDP per capita (current US$)",
    y="Happiness score",
    lowess=True,
)
ax.set_ylabel("Happiness index")
ax.set_title("Is a higher GDP per capita linked to happiness?")
plt.tight_layout()
plt.show()

###############################################################################
# There looks to be a link! Countries with higher GDP per capita tend to
# have higher happiness scores.
#
# Now let's include the legal rights strength table information:

df3 = fuzzy_join(
    left=df2,
    right=legal_rights,
    left_on="Country",
    right_on="Country Name",
    match_score=0.65,
)

###############################################################################
# Let's take a look at the correspondance in a figure:

plt.figure(figsize=(4, 3))
fig = sns.regplot(
    data=df3,
    x="Strength of legal rights index (0=weak to 12=strong)",
    y="Happiness score",
    lowess=True,
)
fig.set_ylabel("Happiness index")
fig.set_title("Does a country's legal rights strength lead to happiness?")
plt.tight_layout()
plt.show()

###############################################################################
# From this plot, it is not clear that this measure of legal strength
# is linked to happiness.
#
# Great! Our joined table has become bigger and full of useful information.
# And now we are ready to apply a first machine learning model to it!

###############################################################################
# Prediction model
# ----------------
#
# We now separate our explanatory variables - X - from our target y.

y = df2[["Happiness score"]]
X = df2.drop("Happiness score", axis=1).select_dtypes(exclude=object)

###################################################################
# Let us now define the model that will be used to predict the happiness score:

from sklearn import __version__ as sklearn_version

if sklearn_version < "1.0":
    from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

hgdb = HistGradientBoostingRegressor(random_state=0)
cv = KFold(n_splits=2, shuffle=True, random_state=0)

#################################################################
# To evaluate our model, we will apply a 4-fold cross-validation,
# and evaluate it using the `R2` score.
#
# Let's finally assess the results of our models:

from sklearn.model_selection import cross_validate

cv_results = cross_validate(hgdb, X, y, cv=cv, scoring="r2")

cv_r2 = cv_results["test_score"]

print(f"Mean R2 score is {cv_r2.mean():.2f} +- {cv_r2.std():.2f}")

#################################################################
# We have a satisfying first result: an R2 of 0.66!
#
# Conclusion
# ----------
#
# Data cleaning varies from dataset to dataset: there are as
# many ways to clean a table as there are errors. The |fuzzy_join|
# method is generalizable across all datasets.
#
# Data transformation is also often very costly in both time and ressources.
# |fuzzy_join| is fast and easy-to-use.
#
# **Now it's up to you: try improving our model by adding relevant
# information into it and beat our result!**
