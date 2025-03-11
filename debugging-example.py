# %%

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

matplotlib.use("QtAgg")

from skrub import DatetimeEncoder, TableVectorizer, datasets

data = datasets.fetch_bike_sharing().bike_sharing

# Extract our input data (X) and the target column (y)
y = data["cnt"]
X = data[["date", "holiday", "temp", "hum", "windspeed", "weathersit"]]

X

# %%
table_vec = TableVectorizer()
X_t = table_vec.fit_transform(X)
# %%
mask_train = X["date"] < "2012-01-01"
X_train, X_test = X.loc[mask_train], X.loc[~mask_train]
y_train, y_test = y.loc[mask_train], y.loc[~mask_train]

# %%

table_vec = TableVectorizer()

table_vec_periodic = TableVectorizer(
    datetime=DatetimeEncoder(
        periodic_encoding="spline",
    )
)

table_vec_weekday = TableVectorizer(
    datetime=DatetimeEncoder(add_weekday=True, add_day_of_year=True)
)


pipeline = make_pipeline(table_vec, HistGradientBoostingRegressor())
pipeline_weekday = make_pipeline(table_vec_weekday, HistGradientBoostingRegressor())
pipeline_periodic = make_pipeline(table_vec_periodic, HistGradientBoostingRegressor())


# pipeline = make_pipeline(table_vec,HistGradientBoostingRegressor())
pipeline.fit(X_train, y_train)
pipeline_weekday.fit(X_train, y_train)
pipeline_periodic.fit(X_train, y_train)

print("hist base: ", pipeline.score(X_test, y_test))
print("hist weekday: ", pipeline_weekday.score(X_test, y_test))
print("hist periodic: ", pipeline_periodic.score(X_test, y_test))


y_pred_base = pipeline.predict(X_test)
y_pred_weekday = pipeline_weekday.predict(X_test)
y_pred_periodic = pipeline_periodic.predict(X_test)

# %%
model = HistGradientBoostingRegressor()

model.fit(X_train, y_train)
model.score(X_test, y_test)
# %%


X_plot = pd.to_datetime(X.tail(1000)["date"]).values


fig, ax = plt.subplots(figsize=(12, 3))
fig.suptitle("Predictions with tree models")
ax.plot(
    X_plot,
    y[-1000:].values,
    "x-",
    alpha=0.2,
    label="Actual demand",
    color="black",
)

ax.plot(
    X_plot,
    y_pred_base[-1000:],
    "x-",
    alpha=0.2,
    label="hist",
    color="blue",
)

ax.plot(
    X_plot,
    y_pred_weekday[-1000:],
    "x-",
    alpha=0.2,
    label="ridge",
    color="green",
)


ax.plot(
    X_plot,
    y_pred_periodic[-1000:],
    "x-",
    alpha=0.2,
    label="ridge",
    color="red",
)

plt.show()
# %%
