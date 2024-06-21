"""
.. _example_datetime_encoder :

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

.. |HGBR| replace::
    :class:`~sklearn.ensemble.HistGradientBoostingRegressor`

.. |to_datetime| replace::
    :func:`~skrub.to_datetime`
"""


###############################################################################
# A problem with relevant datetime features
# -----------------------------------------
#
# We will use a dataset of air quality measurements in different cities.
# In this setting, we want to predict the NO2 air concentration, based
# on the location, date and time of measurement.

from pprint import pprint

import pandas as pd

data = pd.read_csv(
    "https://raw.githubusercontent.com/pandas-dev/pandas"
    "/main/doc/data/air_quality_no2_long.csv"
).sort_values("date.utc")
# Extract our input data (X) and the target column (y)
y = data["value"]
X = data[["city", "date.utc"]]

X

###############################################################################
# We convert the dataframe date columns using |to_datetime|. Notice how
# we don't need to specify the columns to convert.
from skrub import to_datetime

X = to_datetime(X)
X.dtypes

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

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

from skrub import DatetimeEncoder

encoder = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore"), ["city"]),
    (DatetimeEncoder(add_weekday=True, resolution="minute"), "date.utc"),
    remainder="drop",
)

X_enc = encoder.fit_transform(X)
# pprint(encoder.get_feature_names_out())

###############################################################################
# We see that the encoder is working as expected: the ``"date.utc"`` column has
# been replaced by features extracting the month, day, hour, minute, day of the
# week and total second since Epoch information.

###############################################################################
# One-liner with the |TableVectorizer|
# ....................................
#
# As mentioned earlier, the |TableVectorizer| makes use of the
# |DatetimeEncoder| by default.

from skrub import TableVectorizer

table_vec = TableVectorizer().fit(X)
pprint(table_vec.get_feature_names_out())

###############################################################################
# If we want to customize the |DatetimeEncoder| inside the |TableVectorizer|,
# we can replace its default parameter with a new, custom instance:
#
# Here, for example, we want it to extract the day of the week.

table_vec = TableVectorizer(datetime=DatetimeEncoder(add_weekday=True)).fit(X)
pprint(table_vec.get_feature_names_out())

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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

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

mask_train = X["date.utc"] < "2019-06-01"
X_train, X_test = X.loc[mask_train], X.loc[~mask_train]
y_train, y_test = y.loc[mask_train], y.loc[~mask_train]

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

all_cities = X_test["city"].unique()

fig, axes = plt.subplots(nrows=len(all_cities), ncols=1, figsize=(12, 9))
for ax, city in zip(axes, all_cities):
    mask_prediction = X_test["city"] == city
    date_prediction = X_test.loc[mask_prediction]["date.utc"]
    y_prediction = y_pred[mask_prediction]

    mask_reference = X["city"] == city
    date_reference = X.loc[mask_reference]["date.utc"]
    y_reference = y[mask_reference]

    ax.plot(date_reference, y_reference, label="Actual")
    ax.plot(date_prediction, y_prediction, label="Predicted")

    ax.set(
        ylabel="NO2",
        title=city,
    )
    ax.legend()

fig.subplots_adjust(hspace=0.5)
plt.show()

###############################################################################
# Let's zoom on a few days:

mask_zoom_reference = (X["date.utc"] >= "2019-06-01") & (X["date.utc"] < "2019-06-04")
mask_zoom_prediction = (X_test["date.utc"] >= "2019-06-01") & (
    X_test["date.utc"] < "2019-06-04"
)

all_cities = ["Paris", "London"]
fig, axes = plt.subplots(nrows=len(all_cities), ncols=1, figsize=(12, 9))
for ax, city in zip(axes, all_cities):
    mask_prediction = (X_test["city"] == city) & mask_zoom_prediction
    date_prediction = X_test.loc[mask_prediction]["date.utc"]
    y_prediction = y_pred[mask_prediction]

    mask_reference = (X["city"] == city) & mask_zoom_reference
    date_reference = X.loc[mask_reference]["date.utc"]
    y_reference = y[mask_reference]

    ax.plot(date_reference, y_reference, label="Actual")
    ax.plot(date_prediction, y_prediction, label="Predicted")

    ax.set(
        ylabel="NO2",
        title=city,
    )
    ax.legend()

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

table_vec = TableVectorizer(datetime=DatetimeEncoder(add_weekday=True))

# In this case, we don't use a pipeline, because we want to compute the
# importance of the features created by the DatetimeEncoder
X_transform = table_vec.fit_transform(X)
feature_names = table_vec.get_feature_names_out()

model = HistGradientBoostingRegressor().fit(X_transform, y)
result = permutation_importance(model, X_transform, y, n_repeats=10, random_state=0)

result = pd.DataFrame(
    dict(
        feature_names=feature_names,
        std=result.importances_std,
        importances=result.importances_mean,
    )
).sort_values("importances", ascending=False)

result.plot.barh(
    y="importances", x="feature_names", title="Feature Importances", figsize=(12, 9)
)
plt.tight_layout()
plt.show()

# %%
# We can see that the total seconds since Epoch and the hour of the day
# are the most important feature, which seems reasonable.
#
# Conclusion
# ----------
#
# In this example, we saw how to use the |DatetimeEncoder| to create
# features from a date column.
# Also check out the |TableVectorizer|, which automatically recognizes
# and transforms datetime columns by default.
