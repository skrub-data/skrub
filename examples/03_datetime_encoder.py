"""
.. _example_datetime_encoder

===================================================
Handling datetime features with the DatetimeEncoder
===================================================

In this example, we illustrate how to better integrate datetime features
in machine learning models with the |DatetimeEncoder|.

This encoder breaks down passed datetime features into relevant numerical
features, such as the month, the day of the week, the hour of the day, etc.

It is used by default in the |TableVectorizer|.


.. |DatetimeEncoder| replace::
    :class:`~skrub.DatetimeEncoder`

.. |TableVectorizer| replace::
    :class:`~skrub.TableVectorizer`

.. |OneHotEncoder| replace::
    :class:`~sklearn.preprocessing.OneHotEncoder`

.. |TimeSeriesSplit| replace::
    :class:`~sklearn.model_selection.TimeSeriesSplit`

.. |ColumnTransformer| replace::
    :class:`~sklearn.compose.ColumnTransformer`

.. |make_column_transformer| replace::
    :class:`~sklearn.compose.make_column_transformer`

.. |TimeSeriesSplit| replace::
    :class:`~sklearn.model_selection.TimeSeriesSplit`

.. |HGBR| replace::
    :class:`~sklearn.ensemble.HistGradientBoostingRegressor`
"""

import warnings

warnings.filterwarnings("ignore")

###############################################################################
# A problem with relevant datetime features
# -----------------------------------------
#
# We will use a dataset of air quality measurements in different cities.
# In this setting, we want to predict the NO2 air concentration, based
# on the location, date and time of measurement.

import pandas as pd

data = pd.read_csv(
    "https://raw.githubusercontent.com/pandas-dev/pandas"
    "/main/doc/data/air_quality_no2_long.csv"
)
# Extract our input data (X) and the target column (y)
y = data["value"]
X = data[["city", "date.utc"]]

X

###############################################################################
# Encoding the features
# .....................
#
# We will construct a |ColumnTransformer| in which we will encode
# the city names with a |OneHotEncoder|, and the date
# with a |DatetimeEncoder|.
#
# During the instantiation of the |DatetimeEncoder|, we specify that we want
# to extract the day of the week, and that we don't want to extract anything
# finer than minutes. This is because we don't want to extract seconds and
# lower units, as they are probably unimportant.

from sklearn.preprocessing import OneHotEncoder

from skrub import DatetimeEncoder

from sklearn.compose import make_column_transformer

encoder = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore"), ["city"]),
    (DatetimeEncoder(add_day_of_the_week=True, extract_until="minute"), ["date.utc"]),
    remainder="drop",
)

X_enc = encoder.fit_transform(X)
encoder.get_feature_names_out()

###############################################################################
# We see that the encoder is working as expected: the "date.utc" column has
# been replaced by features extracting the month, day, hour, and day of the
# week information.
#
# Note the year and minute features are not present, this is because they
# have been removed by the encoder as they are constant the whole period.

###############################################################################
# One-liner with the |TableVectorizer|
# ....................................
#
# As mentioned earlier, the |TableVectorizer| makes use of the
# |DatetimeEncoder| by default.

from skrub import TableVectorizer
from pprint import pprint

from skrub import TableVectorizer

table_vec = TableVectorizer()
table_vec.fit_transform(X)
pprint(table_vec.get_feature_names_out())

###############################################################################
# If we want to customize the |DatetimeEncoder| inside the |TableVectorizer|,
# we can replace its default parameter with a new, custom instance:
#
# Here, for example, we want it to extract the day of the week.

table_vec = TableVectorizer(
    datetime_transformer=DatetimeEncoder(add_day_of_the_week=True),
)
table_vec.fit_transform(X)
table_vec.get_feature_names_out()

###############################################################################
# .. note:
#     For more information on how to customize the |TableVectorizer|, see
#     :ref:`sphx_glr_auto_examples_01_dirty_categories.py`.
#
# Inspecting the |TableVectorizer| further, we can check that the
# |DatetimeEncoder| is used on the correct column(s).
pprint(table_vec.transformers_)

###############################################################################
# Prediction with datetime features
# ---------------------------------
#
# For prediction tasks, we recommend using the |TableVectorizer| inside a
# pipeline, combined with a model that can use the features extracted by the
# |DatetimeEncoder|.
# Here's we'll use a |HGBR| as our learner.
#
# .. note:
#    You might need to require the experimental feature for scikit-learn
#    versions earlier than 1.0 with:
#    ```py
#    from sklearn.experimental import enable_hist_gradient_boosting
#    ```

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

table_vec = TableVectorizer(
    datetime_transformer=DatetimeEncoder(add_day_of_the_week=True),
)
pipeline = make_pipeline(table_vec, HistGradientBoostingRegressor())

###############################################################################
# Evaluating the model
# ....................
#
# When using date and time features, we often care about predicting the future.
# In this case, we have to be careful when evaluating our model, because
# the standard settings of the cross-validation do not respect time ordering.
#
# Instead, we can use the |TimeSeriesSplit|,
# which ensures that the test set is always in the future.

X["date.utc"] = pd.to_datetime(X["date.utc"])
sorted_indices = np.argsort(X["date.utc"])
X = X.iloc[sorted_indices]
y = y.iloc[sorted_indices]

from sklearn.model_selection import TimeSeriesSplit, cross_val_score

cross_val_score(
    pipeline,
    X,
    y,
    scoring="neg_mean_squared_error",
    cv=TimeSeriesSplit(n_splits=5),
)

###############################################################################
# Plotting the prediction
# .......................
#
# The mean squared error is not obvious to interpret, so we compare
# visually the prediction of our model with the actual values.

import matplotlib.pyplot as plt
from matplotlib.dates import ConciseDateFormatter

X_train = X[X["date.utc"] < "2019-06-01"]
X_test = X[X["date.utc"] >= "2019-06-01"]

y_train = y[X["date.utc"] < "2019-06-01"]
y_test = y[X["date.utc"] >= "2019-06-01"]

pipeline.fit(X_train, y_train)

all_cities = X_test["city"].unique()

fig, axs = plt.subplots(nrows=len(all_cities), ncols=1, figsize=(12, 9))

for i, city in enumerate(all_cities):
    axs[i].plot(
        X.loc[X.city == city, "date.utc"],
        y.loc[X.city == city],
        label="Actual",
    )
    axs[i].plot(
        X_test.loc[X_test.city == city, "date.utc"],
        pipeline.predict(X_test.loc[X_test.city == city]),
        label="Predicted",
    )
    axs[i].set_title(city)
    axs[i].set_ylabel("NO2")
    axs[i].xaxis.set_major_formatter(
        ConciseDateFormatter(axs[i].xaxis.get_major_locator())
    )
    axs[i].legend()
plt.show()

###############################################################################
# Let's zoom on a few days:

X_zoomed = X[X["date.utc"] <= "2019-06-04"][X["date.utc"] >= "2019-06-01"]
y_zoomed = y[X["date.utc"] <= "2019-06-04"][X["date.utc"] >= "2019-06-01"]

X_train_zoomed = X_zoomed[X_zoomed["date.utc"] < "2019-06-03"]
X_test_zoomed = X_zoomed[X_zoomed["date.utc"] >= "2019-06-03"]

y_train_zoomed = y[X["date.utc"] < "2019-06-03"]
y_test_zoomed = y[X["date.utc"] >= "2019-06-03"]

zoomed_cities = X_test_zoomed["city"].unique()

fig, axs = plt.subplots(nrows=len(zoomed_cities), ncols=1, figsize=(12, 9))

for i, city in enumerate(zoomed_cities):
    axs[i].plot(
        X_zoomed.loc[X_zoomed["city"] == city, "date.utc"],
        y_zoomed.loc[X_zoomed["city"] == city],
        label="Actual",
    )
    axs[i].plot(
        X_test_zoomed.loc[X_test_zoomed["city"] == city, "date.utc"],
        pipeline.predict(X_test_zoomed.loc[X_test_zoomed["city"] == city]),
        label="Predicted",
    )
    axs[i].set_title(city)
    axs[i].set_ylabel("NO2")
    axs[i].xaxis.set_major_formatter(
        ConciseDateFormatter(axs[i].xaxis.get_major_locator())
    )
    axs[i].legend()
plt.show()

###############################################################################
# Features importance
# -------------------
#
# Using the |DatetimeEncoder| allows us to better understand how the date
# impacts the NO2 concentration. To this aim, we can compute the
# importance of the features created by the |DatetimeEncoder|, using the
# :func:`~sklearn.inspection.permutation_importance` function, which
# basically shuffles a feature and sees how the model changes its prediction.

###############################################################################
from sklearn.inspection import permutation_importance

table_vec = TableVectorizer(
    datetime_transformer=DatetimeEncoder(add_day_of_the_week=True),
)

# In this case, we don't use a pipeline, because we want to compute the
# importance of the features created by the DatetimeEncoder
X_ = table_vec.fit_transform(X)
reg = HistGradientBoostingRegressor().fit(X_, y)
result = permutation_importance(reg, X_, y, n_repeats=10, random_state=0)
std = result.importances_std
importances = result.importances_mean
indices = np.argsort(importances)
# Sort from least to most
indices = list(reversed(indices))

plt.figure(figsize=(12, 9))
plt.title("Feature importances")
n = len(indices)
labels = np.array(table_vec.get_feature_names_out())[indices]
plt.barh(range(n), importances[indices], color="b", yerr=std[indices])
plt.yticks(range(n), labels, size=15)
plt.tight_layout(pad=1)
plt.show()

###############################################################################
# We can see that the hour of the day is the most important feature,
# which seems reasonable.
#
# Conclusion
# ----------
#
# In this example, we saw how to use the |DatetimeEncoder| to create
# features from a date column.
# Also check out the |TableVectorizer|, which automatically recognizes
# and transforms datetime columns by default.
