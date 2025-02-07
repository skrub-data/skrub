# %%
import pandas as pd

from skrub import DatetimeEncoder, TableVectorizer, datasets

data = datasets.fetch_bike_sharing().bike_sharing

# Extract our input data (X) and the target column (y)
y = data["cnt"]
X = data[["date", "holiday", "temp", "hum", "windspeed", "weathersit"]]

date = pd.to_datetime(X["date"])

# %%
# %%
# %%
de = DatetimeEncoder(add_ordinal_day=True, add_periodic="day")

r = de.fit_transform(date)
print(r.columns)
# %%
tv = TableVectorizer()

# X_t = tv.fit_transform(X)

# %%
