"""
Handling datetime features with the DatetimeEncoder
==========================================

Blabla

"""

###############################################################################
# Data Importing
# --------------
#
# We first get the dataset:
import pandas as pd
from dirty_cat.datasets import fetch_traffic_violations
road_safety = fetch_traffic_violations()
print(road_safety.description)

###############################################################################
# Now, we select relevant features, choose the target and reduce the number of samples
data = road_safety.X[["date_of_stop", "time_of_stop", "longitude", "latitude", "gender", "alcohol", "subagency"]]
y = road_safety.X["accident"] # Whether the traffic violation is linked to an accident
X = road_safety.X.drop("accident", axis=1)
# Reduce dataset size for speed
import numpy as np
rng = np.random.default_rng(1)
indices = rng.choice(range(len(y)), 100)
X, y = X.iloc[indices], y.iloc[indices]

###############################################################################
# Creating encoders for categorical and datetime features
# -------------------------
#
from sklearn.preprocessing import OneHotEncoder
from dirty_cat.datetime_encoder import DatetimeEncoder
cat_encoder = OneHotEncoder(handle_unknown="ignore")
datetime_encoder = DatetimeEncoder(add_day_of_the_week=True)

###############################################################################
# Creating a column encoder
from sklearn.compose import make_column_transformer
datetime_columns = ["date_of_stop", "time_of_stop"]
numerical_columns = ["longitude", "latitude"]
categorical_columns = ["gender", "alcohol", "subagency"]

encoder = make_column_transformer((cat_encoder, categorical_columns),
                                  (datetime_encoder, datetime_columns),
                                  ("passthrough", numerical_columns),
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
# This allows us to compute the importance of these date features, using the permutation_importance function.
from sklearn.ensemble import HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier().fit(X_, y)
from sklearn.inspection import permutation_importance
result = permutation_importance(clf, X_, y, n_repeats=2, random_state=0)
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
# One-liner with the SuperVectorizer
# ----------------------------
# The DatetimeEncoder is used by default in the SuperVectorizer, which automatically detects datetime features.
from dirty_cat import SuperVectorizer
sup_vec = SuperVectorizer()
X_ = sup_vec.fit_transform(X)
feature_names = sup_vec.get_feature_names_out()
feature_names

###############################################################################
# We can see that the SuperVectorizer is indeed using a DatetimeEncoder for the datetime features.
sup_vec.transformers_

###############################################################################
# If we want the day of the week, we can just replace SuperVectorizer's default
sup_vec = SuperVectorizer(datetime_transformer=DatetimeEncoder(add_day_of_the_week=True))
X_ = sup_vec.fit_transform(X)
feature_names = sup_vec.get_feature_names_out()
feature_names
