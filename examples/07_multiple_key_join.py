"""
Joining across multiple columns
===============================

Joining tables may be difficult if one entry on one side does not have
an exact match on the other side.

This problem becomes even more complex when multiple columns
are significant for the join. For instance, this is the case
for **spatial joins** on two columns, typically
longitude and latitude.

|joiner| is a scikit-learn compatible transformer that enables
performing joins across multiple keys,
independantly of the data type (numerical, string or mixed).

The following example uses US domestic flights data
to illustrate how space and time information from a
pool of tables are combined for machine learning.

.. |fj| replace:: :func:`~skrub.fuzzy_join`

.. |joiner| replace:: :func:`~skrub.Joiner`
"""

###############################################################################
# The data
# --------
# The goal is to predict flight delays.
# We have a pool of tables that we will use to improve our prediction.
#
# The following tables are at our disposal:

###############################################################################
# The main table:
#     - The "flights" datasets. It contains all US flights date, origin
#       and destination airports and flight time.
#       Here, we consider only flights from 2008.

import pandas as pd

flights = pd.read_parquet("https://figshare.com/ndownloader/files/41771418")
# Sampling for faster computation.
flights = flights.sample(50_000, random_state=1, ignore_index=True)
flights.head()

############################################################################
# Auxilliary tables provided in the same database:
#     - The "airports" dataset, with information such as their name
#       and location (longitude, latitude).

airports = pd.read_parquet("https://figshare.com/ndownloader/files/41710257")
airports.head()

########################################################################
# Auxilliary tables provided by external sources:
#     - The "stations" dataset. Provides location of all the weather measurement
#       stations in the US.

stations = pd.read_parquet("https://figshare.com/ndownloader/files/41710524")
stations.head()

########################################################################
#     - The "weather" table. Weather details by measurement station.
#       Both tables are from the Global Historical Climatology Network.
#       Here, we consider only weather measurements from 2008.

weather = pd.read_parquet("https://figshare.com/ndownloader/files/41771457")
# Sampling for faster computation.
weather = weather.sample(100_000, random_state=1, ignore_index=True)
weather.head()

###############################################################################
# Joining
# -------
# First we join the stations with weather on the ID (exact join):

aux = pd.merge(stations, weather, on="ID")
aux.head()

###############################################################################
# Then we join this table with the airports so that we get all auxilliary
# tables into one.

from skrub import Joiner

joiner = Joiner(
    tables=[(airports, ["lat", "long"])], main_key=["LATITUDE", "LONGITUDE"]
)

aux_augmented = joiner.fit_transform(aux)

aux_augmented.head()

###############################################################################
# Joining airports with flights data:
# Another multiple key join on the date and the airport

joiner = Joiner(
    tables=[(aux_augmented, ["YEAR/MONTH/DAY", "iata"])],
    main_key=["Year_Month_DayofMonth", "Origin"],
)

main = joiner.fit_transform(flights)
main.head()

###############################################################################
# We now have combined all the information from our pool of tables into one.
# We will use this main table to model the prediction of flight delay.
# Training data is passed through a |Pipeline|:


from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline

tv = TableVectorizer()
hgb = HistGradientBoostingClassifier()

pipeline_hgb = make_pipeline(tv, hgb)

###############################################################################
# We isolate our target variable and remove useless ID variables:

y = main["ArrDelay"]
X = main.drop(columns=["ArrDelay", "FlightNum", "TailNum", "ID", "iata"])

###############################################################################
# We want to frame this as a classification problem:
# suppose that your company is obliged to reimburse the ticket
# price if the flight is delayed.
# We have a binary classification problem:
# the flight was delayed (1) or not (0).

y = (y > 0).astype(int)
y.value_counts()

###############################################################################
# The results:

from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline_hgb, X, y)
scores.mean()

###############################################################################
# Conclusion
# ----------
#
# In this example, we have combined multiple tables with complex joins
# on imprecise and multiple-key correspondences.
# This is made easy by skrub's |Joiner| transformer.
#
# Our final cross-validated accuracy score is 0.6.
