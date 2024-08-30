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

.. |ToDatetime| replace::
    :class:`~skrub.ToDatetime`
"""

###############################################################################
# A problem with relevant datetime features
# -----------------------------------------
#
# We will use a dataset of bike sharing demand in 2011 and 2012.
# In this setting, we want to predict the number of bike rentals, based
# on the date, time and weather conditions.

from pprint import pprint

import pandas as pd

data = pd.read_csv(
    "https://raw.githubusercontent.com/skrub-data/datasets/master"
    "/data/bike-sharing-dataset.csv"
)
# Extract our input data (X) and the target column (y)
y = data["cnt"]
X = data[["date", "holiday", "temp", "hum", "windspeed", "weathersit"]]

X

# %%
y

###############################################################################
# We convert the dataframe's ``"date"`` column using |ToDatetime|.

from skrub import ToDatetime

date = ToDatetime().fit_transform(X["date"])

print("original dtype:", X["date"].dtypes, "\n\nconverted dtype:", date.dtypes)

###############################################################################
# Encoding the features
# .....................
#
# We now encode this column with a |DatetimeEncoder|.
#
# During the instantiation of the |DatetimeEncoder|, we specify that we want
# to extract the day of the week, and that we don't want to extract anything
# finer than hours. This is because we don't want to extract minutes, seconds
# and lower units, as they are unimportant.

from skrub import DatetimeEncoder

date_enc = DatetimeEncoder().fit_transform(date)

print(date, "\n\nHas been encoded as:\n\n", date_enc)

###############################################################################
# We see that the encoder is working as expected: the column has
# been replaced by features extracting the month, day, hour, day of the
# week and total seconds since Epoch information.

###############################################################################
# One-liner with the TableVectorizer
# ..................................
#
# As mentioned earlier, the |TableVectorizer| makes use of the
# |DatetimeEncoder| by default. Note that ``X["date"]`` is still
# a string, but will be automatically transformed into a datetime in the
# |TableVectorizer|.

from skrub import TableVectorizer

table_vec = TableVectorizer().fit(X)
pprint(table_vec.get_feature_names_out())

###############################################################################
# If we want to customize the |DatetimeEncoder| inside the |TableVectorizer|,
# we can replace its default parameter with a new, custom instance.
#
# Here, for example, we want it to extract the day of the week:

# use the ``datetime`` argument to customize how datetimes are handled
table_vec_weekday = TableVectorizer(datetime=DatetimeEncoder(add_weekday=True)).fit(X)
pprint(table_vec_weekday.get_feature_names_out())

###############################################################################
# .. note:
#     For more information on how to customize the |TableVectorizer|, see
#     :ref:`sphx_glr_auto_examples_01_dirty_categories.py`.
#
# Inspecting the |TableVectorizer| further, we can check that the
# |DatetimeEncoder| is used on the correct column(s).
pprint(table_vec_weekday.transformers_)

###############################################################################
# Prediction with datetime features
# ---------------------------------
#
# For prediction tasks, we recommend using the |TableVectorizer| inside a
# pipeline, combined with a model that can use the features extracted by the
# |DatetimeEncoder|.
# Here we'll use a |HGBR| as our learner.
#
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(table_vec, HistGradientBoostingRegressor())
pipeline_weekday = make_pipeline(table_vec_weekday, HistGradientBoostingRegressor())

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
# The mean squared error is not obvious to interpret, so we visually
# compare the prediction of our model with the actual values.
# To do so, we will divide our dataset into a train and a test set:
# we use 2011 data to predict what happened in 2012.
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

mask_train = X["date"] < "2012-01-01"
X_train, X_test = X.loc[mask_train], X.loc[~mask_train]
y_train, y_test = y.loc[mask_train], y.loc[~mask_train]

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

pipeline_weekday.fit(X_train, y_train)
y_pred_weekday = pipeline_weekday.predict(X_test)

fig, ax = plt.subplots(figsize=(12, 3))
fig.suptitle("Predictions with tree models")
ax.plot(
    X.tail(96)["date"],
    y.tail(96).values,
    "x-",
    alpha=0.2,
    label="Actual demand",
    color="black",
)
ax.plot(
    X_test.tail(96)["date"],
    y_pred[-96:],
    "x-",
    label="DatetimeEncoder() + HGBR prediction",
)
ax.plot(
    X_test.tail(96)["date"],
    y_pred_weekday[-96:],
    "x-",
    label="DatetimeEncoder(add_weekday=True) + HGBR prediction",
)

ax.tick_params(axis="x", labelsize=7, labelrotation=75)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
_ = ax.legend()
plt.tight_layout()
plt.show()
###############################################################################
# As we can see, adding the weekday yields better predictions on our test set.


###############################################################################
# Feature importances
# -------------------
#
# Using the |DatetimeEncoder| allows us to better understand how the date
# impacts the bike sharing demand. To this aim, we can compute the
# importance of the features created by the |DatetimeEncoder|, using the
# :func:`~sklearn.inspection.permutation_importance` function, which
# basically shuffles a feature and sees how the model changes its prediction.

###############################################################################
from sklearn.inspection import permutation_importance

# In this case, we don't use a pipeline, because we want to compute the
# importance of the features created by the DatetimeEncoder
X_test_transform = pipeline[:-1].transform(X_test)

result = permutation_importance(
    pipeline[-1], X_test_transform, y_test, n_repeats=10, random_state=0
)

result = pd.DataFrame(
    dict(
        feature_names=X_test_transform.columns,
        std=result.importances_std,
        importances=result.importances_mean,
    )
).sort_values("importances", ascending=True)

result.plot.barh(
    y="importances",
    x="feature_names",
    title="Feature Importances",
    xerr="std",
    figsize=(12, 9),
)
plt.tight_layout()
plt.show()

# %%
# We can see that the hour of the day, the temperature and the humidity
# are the most important features, which seems reasonable.
#
# Conclusion
# ----------
#
# In this example, we saw how to use the |DatetimeEncoder| to create
# features from a datetime column.
# Also check out the |TableVectorizer|, which automatically recognizes
# and transforms datetime columns by default.
