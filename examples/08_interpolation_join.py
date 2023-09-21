"""
Interpolation join: infer missing rows when joining 2 tables
============================================================

In this example we show an interpolation join where the ground truth is known.
To do so, we split a table containing wether data in half and then join both
halves, using the latitude, longitude and date of the weather measurements.
"""

######################################################################
# Load weather data
# -----------------
from skrub.datasets import fetch_figshare
import pandas as pd

weather = fetch_figshare("41771457").X
weather = weather.sample(100_000, random_state=0, ignore_index=True)
stations = fetch_figshare("41710524").X
weather = pd.merge(stations, weather, on="ID").loc[
    :, ["LATITUDE", "LONGITUDE", "YEAR/MONTH/DAY", "TMAX", "PRCP", "SNOW"]
]

n_left = weather.shape[0] // 2


######################################################################
# Split the table
left_table = weather.iloc[:n_left]
left_table = left_table.rename(
    columns={c: f"{c}_true" for c in ["TMAX", "PRCP", "SNOW"]}
)
left_table.head()

######################################################################
right_table = weather.iloc[n_left:]
right_table.head()


######################################################################
# Joining the tables
# ------------------

from skrub import InterpolationJoin

interpolation_join = InterpolationJoin(
    right_table, on=["LATITUDE", "LONGITUDE", "YEAR/MONTH/DAY"]
).fit()
joined = interpolation_join.transform(left_table)
joined.head()

######################################################################
# Comparing the estimated values to the ground truth
# --------------------------------------------------

from matplotlib import pyplot as plt

joined = joined.sample(2000, random_state=0)
for col in ["TMAX", "PRCP", "SNOW"]:
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.scatter(
        joined[f"{col}_true"].values,
        joined[col].values,
        alpha=0.1,
    )
    ax.set_aspect(1)
    ax.set_xlabel(f"true {col}")
    ax.set_ylabel(f"interpolated {col}")
    fig.tight_layout()

######################################################################
# We see that in this case the interpolation join works well for the
# temperature, but not precipitation nor snow.
