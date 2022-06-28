"""
Handling datetime features with the DatetimeEncoder
==========================================

We illustrate here how to handle datetime features with the DatetimeEncoder. The DatetimeEncoder
breaks down each datetime features into several numerical features, by extracting relevant information from the
datetime features, such as the month, the day of the week, the hour of the day, etc. Used in
the SuperVectorizer, which automatically detects the datetime features, the DatetimeEncoder allows
to handle datetime features easily.
"""

###############################################################################
# Data Importing
# --------------
#
# We first get the dataset.
# We want to predict the NO2 air concentration in different cities, based on the date and the time of measurement.
import pandas as pd
data = pd.read_csv("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/air_quality_no2_long.csv")
y = data["value"]
X = data[["city", "date.utc"]]
X

###############################################################################
# Creating encoders for categorical and datetime features
# -------------------------
from sklearn.preprocessing import OneHotEncoder
from dirty_cat.datetime_encoder import DatetimeEncoder

cat_encoder = OneHotEncoder(handle_unknown="ignore")
datetime_encoder = DatetimeEncoder(add_day_of_the_week=True,  # The day of the week is probably relevant
                                   extract_until="minute")  # We're probably not interested in seconds and below

from sklearn.compose import make_column_transformer

datetime_columns = ["date.utc"]
categorical_columns = ["city"]

encoder = make_column_transformer((cat_encoder, categorical_columns),
                                  (datetime_encoder, datetime_columns),
                                  remainder="drop")


###############################################################################
# Transforming the input data
# ----------------------------
# We can see that the encoder is working as expected, and new date features are created.
X_ = encoder.fit_transform(X)
feature_names = encoder.get_feature_names_out()
feature_names

###############################################################################
# Features importance
# ----------------------------
# This allows us to compute the importance of these date features, using the `permutation_importance` function.
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np

clf = HistGradientBoostingRegressor().fit(X_, y)
from sklearn.inspection import permutation_importance

result = permutation_importance(clf, X_, y, n_repeats=10, random_state=0)
std = result.importances_std
importances = result.importances_mean
indices = np.argsort(importances)
# Sort from least to most
indices = list(reversed(indices))
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 9))
plt.title("Feature importances")
n = len(indices)
labels = np.array(feature_names)[indices]
plt.barh(range(n), importances[indices], color="b", yerr=std[indices])
plt.yticks(range(n), labels, size=15)
plt.tight_layout(pad=1)
plt.show()

###############################################################################
# We can see that the hour of the day is the most important feature, which seems reasonable.
# Note that the `year` and `minute` features were removed because they were constant.

###############################################################################
# One-liner with the SuperVectorizer
# ----------------------------
# The DatetimeEncoder is used by default in the SuperVectorizer, which automatically detects datetime features.
from dirty_cat import SuperVectorizer

sup_vec = SuperVectorizer()
X_ = sup_vec.fit_transform(X)
feature_names = sup_vec.get_feature_names_out()
print(feature_names)

###############################################################################
# We can see that the SuperVectorizer is indeed using a DatetimeEncoder for the datetime features.
sup_vec.transformers_

###############################################################################
# If we want the day of the week, we can just replace SuperVectorizer's default
sup_vec = SuperVectorizer(datetime_transformer=DatetimeEncoder(add_day_of_the_week=True))
X_ = sup_vec.fit_transform(X)
feature_names = sup_vec.get_feature_names_out()
print(feature_names)
