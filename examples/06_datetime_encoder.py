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

###############################################################################
# Predicting with date features
# --------------------------------
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(sup_vec, clf)
# When using date features, we often care about predicting the future.
# In this case, we have to be careful when evaluating our model, because
# standard tool like cross-validation do not respect the time ordering.
X.loc[:, "date.utc"] = pd.to_datetime(X["date.utc"])
X.sort_values("date.utc", inplace=True)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
cross_val_score(pipeline, X, y, scoring="neg_mean_squared_error", cv=TimeSeriesSplit(n_splits=5))

###############################################################################
# Plotting the prediction
X_train, X_test = X[X["date.utc"] < "2019-06-01"], X[X["date.utc"] >= "2019-06-01"]
y_train, y_test = y[X["date.utc"] < "2019-06-01"], y[X["date.utc"] >= "2019-06-01"]
pipeline.fit(X_train, y_train)
# Plot subplots for each city using subplots
fig, axs = plt.subplots(nrows=len(X_test.city.unique()), ncols=1, figsize=(12, 9))
for i, city in enumerate(X_test.city.unique()):
    print(X_test.city == city)
    print(y_test)
    axs[i].plot(X.loc[X.city == city, "date.utc"], y.loc[X.city == city], label="Actual")
    axs[i].plot(X_test.loc[X_test.city == city, "date.utc"], pipeline.predict(X_test.loc[X_test.city == city]),
                label="Predicted")
    axs[i].set_title(city)
    axs[i].set_ylabel("NO2")
plt.legend()
plt.show()

###############################################################################
# Let's zoom on a few days

X_zoomed = X[X["date.utc"] <= "2019-06-04"][X["date.utc"] >= "2019-06-01"]
y_zoomed = y[X["date.utc"] <= "2019-06-04"][X["date.utc"] >= "2019-06-01"]
X_train_zoomed, X_test_zoomed = X_zoomed[X_zoomed["date.utc"] < "2019-06-03"], X_zoomed[X_zoomed["date.utc"] >= "2019-06-03"]
y_train_zoomed, y_test_zoomed = y[X["date.utc"] < "2019-06-03"], y[X["date.utc"] >= "2019-06-03"]
pipeline.fit(X_train, y_train)
# Plot subplots for each city using subplots
fig, axs = plt.subplots(nrows=len(X_test_zoomed.city.unique()), ncols=1, figsize=(12, 9))
for i, city in enumerate(X_test_zoomed.city.unique()):
    axs[i].plot(X_zoomed.loc[X_zoomed.city == city, "date.utc"], y_zoomed.loc[X_zoomed.city == city], label="Actual")
    axs[i].plot(X_test_zoomed.loc[X_test_zoomed.city == city, "date.utc"], pipeline.predict(X_test_zoomed.loc[X_test_zoomed.city == city]),
                label="Predicted")
    axs[i].set_title(city)
    axs[i].set_ylabel("NO2")
plt.legend()
plt.show()
