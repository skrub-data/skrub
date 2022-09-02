"""
Joining tables with fuzzyjoin
================================

In this example, we show how to join tables with the :function:`fuzzyjoin` function. 
We also demonstrate why this method is the most easy and appropriate tool for handling 
the joining of tables for users that want to improve their machine learning models quickly.
We go through some of the rich options that allow for manipulation when joining tables,
such as taking into account the precision of the match.

To do so, we will predict the happiness score of a country from
the 2022 [World Happiness Report](https://worldhappiness.report/).
We will also use data provided from the [World Bank open data platform](https://data.worldbank.org/)
in order to create a satisfying prediction model.

"""

###############################################################################
# Data Importing and preprocessing
# --------------------------------
#
# We import the happiness score data first:
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/jovan-stojanovic/data/main/Happiness_report_2022.csv', thousands=',')
df.drop(df.tail(1).index, inplace = True)

# Let's take a look at the table:
df.head(3)
 
#################################################################
# The Happiness score was computed using the Gallup World Poll survey results. 
# The report stress out some of the possible explanatory factors: GDP per capita, Social support, Generosity etc.
# However, these factors here are only estimated indexes used to calculate the happiness score.

# The sum of all explanatory indexes is then the happiness score itself:
df['Sum_of_factors'] = df.iloc[:,[5,6,7,8,9,10,11]].sum(axis=1)
df[['Happiness score','Sum_of_factors']].head(3)

# Thus, we cannot use them for our prediction model.
X = df[['Country']]
y = df[['Happiness score']]

# If we want to create a machine learning model which predicts the happiness index of any new country or future date,
# we will need to include explanatory factors from other tables.

###############################################################################
# Joining tables using fuzzyjoin
# =================================
#
# Finding additional tables
# ---------------------------

# Let's inspire ourselfes from the factors used by the Happiness report to explain happiness. 
# We will extract data from the World Bank databank using the following function:
from dirty_cat.datasets import fetch_world_bank_data

# We then extract GDP per capita by country:
gdppc = fetch_world_bank_data('NY.GDP.PCAP.CD', 'gdppc')
gdppc.head(3)

# Life expectancy by country:
life_exp = fetch_world_bank_data('SP.DYN.LE00.IN', indicator='life_exp')
life_exp.head(3)

# And the legal rights strength by country:
legal_rights = fetch_world_bank_data('IC.LGL.CRED.XQ', indicator='legal_rights')
legal_rights.head(3)


# Joining tables
# -----------------------
#
# Now, using dirty_cat's fuzzyjoin function,
# we need only one line to join two tables
# without worrying about preprocessing:

# We add GDP per capita to the initial table:
from dirty_cat._fuzzy_join import fuzzyjoin
X1 = fuzzyjoin(X, gdppc, on=['Country', 'Country Name'])
X1.head(20)
#################################################################

# Now, we see that our fuzzyjoin succesfully identified the countries,
# even though some country names differ between tables. 

# For instance, Czechia is well identified as Czech Republic and Luxembourg* as Luxembourg. 

# This would all be missed out if we were using other methods such as
# [pandas.DataFrame.join](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html),
# which searches only for exact matches.
# In this case, to reach the best result, we would have to manually clean
# the data (e.g. remove the * after country name) and look manually
# for matching patterns in observations. 

# Dirty_cat's fuzzyjoin is the perfect function to avoid doing so (and save time) with great results.

# However, we see that some matches were unsuccesful (e.g 'Palestinian Territories*' and 'Timor-Leste'),
# because there is simply no match in the two tables.
X1.iloc[121]
#################################################################

# In this case, it is better to use the '2dball' precision
# with a fixed threshold so as to include only precise-enough matches:

# --> To improve precision measurement, here it excludes some good matches as well
X1 = fuzzyjoin(X, gdppc, on=['Country', 'Country Name'], precision='2dball', precision_threshold=0.3)
X1.iloc[121]
# Matches that are not available (or precise enough) are thus marked as `NaN`.
#################################################################

# Now let's include other information that may be relevant, such as life expectancy:
X2 = fuzzyjoin(X1, life_exp,  on=['Country', 'Country Name'], precision='2dball', precision_threshold=0.3, keep='left')
X2.head(3)
#################################################################

# Note: Here, we use the `keep='left'` option to keep only the left key matching column,
# so as not to have too much unnecessary columns with country names.

# And the strenght of legal rights in the country:
X3 = fuzzyjoin(X2, legal_rights,  on=['Country', 'Country Name'], precision='2dball', precision_threshold=0.3, keep='left')
X3.head(3)
#################################################################

# Great! Our table has became bigger and full of useful informations.
# We now only remove categories with missing information:
import numpy as np
mask = X3['gdppc'].notna()
y = np.ravel(y[mask])

X1 = X1[mask]
X2 = X2[mask]
X3 = X3[mask]

# And we are ready to apply a machine learning model to it!

###############################################################################
# Prediction model
# ==============================
#

# Let us now define the model that will be used to predict the happiness score:
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate, KFold
hgdb = HistGradientBoostingRegressor(random_state=0)
cv = KFold(n_splits=4, shuffle=True, random_state=0)

# To evaluate our model, we will apply a `4-fold cross-validation`. 
# We evaluate our model using the `R2` score.

# Let's finally assess the results of our models:
for n in range(len([X1, X2, X3])):
    data = [X1, X2, X3][n] 
    cv_results_t = cross_validate(hgdb, data.select_dtypes(exclude=object), y, cv=cv, scoring='r2')
    cv_r2_t = cv_results_t['test_score']
    print(f'Mean R2 score with table {n+1} is {cv_r2_t.mean():.2f} +- {cv_r2_t.std():.2f}')

# Our score gets better every time we add additional information into our table !

# This is why dirty_cat's FuzzyJoin is an easy-to-use
# and useful tool.
