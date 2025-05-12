"""
A script for generating a plot that displays how periodic features are encoded,
to be added to the front page.
"""

import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import polars as pl
from polars import selectors as cs

from skrub import DatetimeEncoder

print("Generating periodic encoding example for the main page...")

# Save HTML snippets of the tablesdd
OUTPUT_DIR = pathlib.Path("generated_for_index")
# Create the output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Preparing first dataframe with smaller resolution
df_1 = pl.datetime_range(
    datetime(2022, 1, 1), datetime(2023, 1, 1), interval="6h", eager=True
).alias("Datetime")
de = DatetimeEncoder(periodic_encoding="spline", add_weekday=True)
X = de.fit_transform(df_1)

day = X.select(cs.starts_with("Datetime_day")).to_numpy()
week = X.select(cs.starts_with("Datetime_weekday")).to_numpy()
hour = X.select(cs.starts_with("Datetime_hour")).to_numpy()

df_2 = pl.datetime_range(
    datetime(2022, 1, 1), datetime(2023, 1, 1), interval="15d", eager=True
).alias("Datetime")
de = DatetimeEncoder(periodic_encoding="spline", add_weekday=True, add_day_of_year=True)
X = de.fit_transform(df_2)
month = X.select(cs.starts_with("Datetime_month")).to_numpy()
doy = X.select(cs.starts_with("Datetime_day_of_year")).to_numpy()


# Add datetime-based x-ticks to the figure
fig, axs = plt.subplots(1, 5, figsize=(7, 4), layout="constrained")
datetimes = df_1.to_pandas()[:48]
axs[0].imshow(hour[:48], aspect=1.5)
axs[1].imshow(week[:48], aspect=1.5)
axs[2].imshow(day[:48], aspect=1.5)

tick_positions = range(
    0, len(datetimes), 4
)  # Show a tick every 4th value (adjust as needed)
tick_labels = [
    d.strftime("%Y-%m-%d") for i, d in enumerate(datetimes) if i % 4 == 0
]  # Format as days only

for ax in axs[:3]:
    ax.set_yticks(tick_positions)  # Set tick positions
    ax.set_yticklabels(
        tick_labels, rotation=45, ha="right", fontsize=7
    )  # Set tick labels
    ax.set_xticks([])

axs[0].set_title("Hours in day")
axs[1].set_title("Day in week")
axs[2].set_title("Day in month")

axs[3].imshow(month, aspect=4)
axs[4].imshow(doy, aspect=4)

axs[3].set_title("Month in year")
axs[4].set_title("Day in year")
datetimes = df_2.to_pandas()
tick_positions = range(0, len(datetimes), 4)  # Show a tick every 4th value
tick_labels = [
    d.strftime("%Y-%m-%d") for i, d in enumerate(datetimes) if i % 4 == 0
]  # Format as days only

for ax in axs[3:]:
    ax.set_yticks(tick_positions)  # Set tick positions
    ax.set_yticklabels(
        tick_labels, rotation=45, ha="right", fontsize=7
    )  # Set tick labels
    ax.set_xticks([])

fig.supxlabel("Spline features")

fig.savefig(OUTPUT_DIR / "demo_periodic_features.svg")
