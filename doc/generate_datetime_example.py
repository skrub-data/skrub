"""
A script for generating a plot that displays how periodic features are encoded,
to be added to the front page.
"""

import datetime as dt
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import polars as pl
from polars import selectors as cs

from skrub import DatetimeEncoder

print("Generating periodic encoding example for the main page...")

# Save HTML snippets of the tablesdd
OUTPUT_DIR = pathlib.Path("generated_for_index")
# Create the output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 14})  # Set to your desired font size
# %%
# Defining a separate dataframe for each range for better control on the x-axis ticks

df_day = pl.datetime_range(
    datetime(2022, 1, 1), datetime(2022, 1, 3), interval="30m", eager=True
).alias("Datetime")
de = DatetimeEncoder(periodic_encoding="spline", add_weekday=True)
X = de.fit_transform(df_day)
hour = X.select(cs.starts_with("Datetime_hour")).to_numpy()

df_week = pl.datetime_range(
    datetime(2022, 1, 1), datetime(2022, 1, 8), interval="1h", eager=True
).alias("Datetime")
de = DatetimeEncoder(periodic_encoding="spline", add_weekday=True)
X = de.fit_transform(df_week)
week = X.select(cs.starts_with("Datetime_weekday")).to_numpy()

df_month = pl.datetime_range(
    datetime(2022, 1, 1), datetime(2022, 2, 1), interval="12h", eager=True
).alias("Datetime")
de = DatetimeEncoder(periodic_encoding="spline", add_weekday=True)
X = de.fit_transform(df_month)
day = X.select(cs.starts_with("Datetime_day")).to_numpy()

df_year = pl.datetime_range(
    datetime(2022, 1, 1), datetime(2023, 1, 1), interval="1d", eager=True
).alias("Datetime")
de = DatetimeEncoder(periodic_encoding="spline")
X = de.fit_transform(df_year)
month = X.select(cs.starts_with("Datetime_month")).to_numpy()


# %%
def hour_tick_formatter(x, pos):
    # x is the tick position (index)
    base_time = datetime(2022, 1, 1)
    delta = base_time + dt.timedelta(minutes=30 * int(x))
    return delta.strftime("%H:%M")


def half_hour_tick_formatter(x, pos):
    # x is the tick position (index)
    base_time = datetime(2022, 1, 1)
    delta = base_time + dt.timedelta(minutes=30 * int(x))
    return delta.strftime("%A")


def day_tick_formatter(x, pos):
    # x is the tick position (index)
    base_time = datetime(2022, 1, 1)
    delta = base_time + dt.timedelta(minutes=60 * int(x))
    return delta.strftime("%A")


def day_in_month_tick_formatter(x, pos):
    # x is the tick position (index)
    base_time = datetime(2022, 1, 1)
    delta = base_time + dt.timedelta(days=int(x))
    return delta.strftime("%a,\n%d %B")


def month_tick_formatter(x, pos):
    # x is the tick position (index)
    base_time = datetime(2022, 1, 1)
    delta = base_time + dt.timedelta(weeks=4 * int(x))
    return delta.strftime("%a,\n%d %B")


def first_of_month_tick_formatter(x, pos):
    # Returns the month name for the first day of each month
    # This avoids having to keep track of the number of days in each month
    base_time = datetime(2022, 1, 1)
    date = base_time + dt.timedelta(days=int(x))
    return date.strftime("%B") if date.day == 1 else ""


fig, axs = plt.subplots(
    4,
    1,
    figsize=(8, 6),
    layout="constrained",
)

ax_hour = axs[0]
ax_week = axs[1]
ax_month = axs[2]
ax_year = axs[3]

# Plot for hours in a day
ax_hour.imshow(hour.T, aspect="auto")
major_tick_positions = range(0, 48 * 2, 48)
minor_tick_positions = range(0, 48 * 2, 12)
ax_hour.set_xticks(major_tick_positions)
ax_hour.set_xticks(minor_tick_positions, minor=True)
ax_hour.xaxis.set_minor_formatter(ticker.FuncFormatter(hour_tick_formatter))
ax_hour.xaxis.set_major_formatter(ticker.FuncFormatter(half_hour_tick_formatter))
ax_hour.set_title("Hour in day")
ax_hour.set_yticks([])

# Plot for weekdays in a week
ax_week.imshow(week[: 24 * 7].T, aspect="auto")
major_tick_positions = range(0, 24 * 7, 24)
ax_week.set_xticks(major_tick_positions)
ax_week.xaxis.set_major_formatter(ticker.FuncFormatter(day_tick_formatter))
ax_week.set_title("Weekday")
ax_week.set_yticks([])

# Plot for days in a month
ax_month.imshow(day[:].T, aspect="auto")
major_tick_positions = range(0, 2 * 31, 7 * 2)
ax_month.set_xticks(major_tick_positions)
ax_month.xaxis.set_major_formatter(ticker.FuncFormatter(day_in_month_tick_formatter))
ax_month.set_title("Day in month")
ax_month.set_yticks([])

# Plot for months in a year
ax_year.imshow(month[:].T, aspect="auto")
first_days = []
base_time = datetime(2022, 1, 1)
for i in range((datetime(2023, 1, 1) - base_time).days):
    date = base_time + dt.timedelta(days=i)
    if date.day == 1:
        first_days.append(i)

ax_year.set_xticks(first_days)
plt.setp(ax_year.get_xticklabels(), rotation=45)
ax_year.xaxis.set_major_formatter(ticker.FuncFormatter(first_of_month_tick_formatter))
ax_year.set_title("Month in year")
ax_year.set_yticks([])

fig.savefig(OUTPUT_DIR / "demo_periodic_features.svg")
