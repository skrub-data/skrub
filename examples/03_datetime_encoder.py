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
# the date with a |DatetimeEncoder|.
#
# During the instantiation of the |DatetimeEncoder|, we specify that we want
# to extract the day of the week, and that we don't want to extract anything
# finer than hours. This is because we don't want to extract minutes, seconds
# and lower units, as they are unimportant.

from sklearn.compose import make_column_transformer

from skrub import DatetimeEncoder

encoder = make_column_transformer(
    (DatetimeEncoder(add_weekday=True, resolution="hour"), "date"),
    remainder="drop",
)

X_enc = encoder.fit_transform(X)

X_enc

###############################################################################
# We see that the encoder is working as expected: the ``"date"`` column has
# been replaced by features extracting the month, day, hour, day of the
# week and total seconds since Epoch information.

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

table_vec_wd = TableVectorizer(datetime=DatetimeEncoder(add_weekday=True)).fit(X)
pprint(table_vec_wd.get_feature_names_out())

###############################################################################
# .. note:
#     For more information on how to customize the |TableVectorizer|, see
#     :ref:`sphx_glr_auto_examples_01_dirty_categories.py`.
#
# Inspecting the |TableVectorizer| further, we can check that the
# |DatetimeEncoder| is used on the correct column(s).
pprint(table_vec_wd.transformers_)

###############################################################################
# Prediction with datetime features
# ---------------------------------
#
# For prediction tasks, we recommend using the |TableVectorizer| inside a
# pipeline, combined with a model that can use the features extracted by the
# |DatetimeEncoder|.
# Here's we'll use a |HGBR| as our learner.
#
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(table_vec, HistGradientBoostingRegressor())
pipeline_wd = make_pipeline(table_vec_wd, HistGradientBoostingRegressor())

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

mask_train = X["date"] < "2012-01-01"
X_train, X_test = X.loc[mask_train], X.loc[~mask_train]
y_train, y_test = y.loc[mask_train], y.loc[~mask_train]

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

pipeline_wd.fit(X_train, y_train)
y_pred_wd = pipeline_wd.predict(X_test)

fig, ax = plt.subplots(figsize=(12, 3))
fig.suptitle("Predictions by linear models")
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
    label="DatetimeEncoder() + HGBD prediction",
)
ax.plot(
    X_test.tail(96)["date"],
    y_pred_wd[-96:],
    "x-",
    label="DatetimeEncoder(add_weekday=True) + HGBD prediction",
)

_ = ax.legend()
plt.show()


###############################################################################
# Features importance
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
).sort_values("importances", ascending=False)

result.plot.barh(
    y="importances", x="feature_names", title="Feature Importances", figsize=(12, 9)
)
plt.tight_layout()
plt.show()

# %%
# We can see that the hour of the day, the temperature and the humidity
# are the most important feature, which seems reasonable.
#
# Conclusion
# ----------
#
# In this example, we saw how to use the |DatetimeEncoder| to create
# features from a date column.
# Also check out the |TableVectorizer|, which automatically recognizes
# and transforms datetime columns by default.
