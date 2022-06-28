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
from dirty_cat.datasets import fetch_traffic_violations

road_safety = fetch_traffic_violations()
print(road_safety.description)

###############################################################################
# Now, we select relevant features, choose the target and reduce the number of samples.
# We want to predict whether a traffic violation is linked to an accident, depending on the location, the gender
# of the driver, whether the driver had drunk alcohol, and, most importantly, the date and time of the violation.
data = road_safety.X[["accident", "date_of_stop", "time_of_stop", "longitude", "latitude", "gender", "alcohol", "subagency"]]
import gc #reduce memory usage
del road_safety
gc.collect()
y = data["accident"]  # Whether the traffic violation is linked to an accident
X = data.drop("accident", axis=1)
# Reduce dataset size for speed
import numpy as np
rng = np.random.default_rng(1)
indices = rng.choice(range(len(y)), 50000)
X, y = X.iloc[indices], y.iloc[indices]

###############################################################################
# Creating encoders for categorical and datetime features
# -------------------------
from sklearn.preprocessing import OneHotEncoder
from dirty_cat.datetime_encoder import DatetimeEncoder

cat_encoder = OneHotEncoder(handle_unknown="ignore")
datetime_encoder = DatetimeEncoder(add_day_of_the_week=True,  # The day of the week is probably relevant
                                   extract_until="hour")  # We're probably not interested in minutes, seconds etc.

from sklearn.compose import make_column_transformer

datetime_columns = ["date_of_stop", "time_of_stop"]
numerical_columns = ["longitude", "latitude"]
categorical_columns = ["gender", "alcohol", "subagency"]

encoder = make_column_transformer((cat_encoder, categorical_columns),
                                  (datetime_encoder, datetime_columns),
                                  ("passthrough", numerical_columns),  # don't encode the numerical columns
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
from sklearn.ensemble import HistGradientBoostingClassifier

clf = HistGradientBoostingClassifier().fit(X_, y)
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
# We can see that datetime features are considered important by the model.
# The "full" feature, which represent the full time to epoch, seems the most
# important (note that it is highly collinear with the `year` feature, which can
# make the importance interpretation tricky). The other features seem approximately
# equally important, with the day of the month being slightly more important.

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
