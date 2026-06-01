"""
This script is used to generate the plots that show the periodic features generated
by the DatetimeEncoder that is shown in the home page of the website.

The plot was added in https://github.com/skrub-data/skrub/pull/1924

# colormap used in this plot is from:
Thyng, K. M., Greene, C. A., Hetland, R. D., Zimmerle, H. M., & DiMarco, S. F. (2016).
True colors of oceanography. Oceanography, 29(3), 10.

To reproduce, install the cmocean package and use the following colormap:
cmap = cmocean.cm.thermal

link: http://tos.org/oceanography/assets/docs/29-3_thyng.pdf
"""

# %%
import datetime as dt
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import polars as pl
from polars import selectors as cs

from skrub import DatetimeEncoder

plt.rcParams.update({"font.size": 20})  # Set to your desired font size
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
day = X.select(cs.starts_with("Datetime_weekday")).to_numpy()

df_month = pl.datetime_range(
    datetime(2022, 1, 1), datetime(2022, 2, 1), interval="12h", eager=True
).alias("Datetime")
de = DatetimeEncoder(periodic_encoding="spline", add_weekday=True)
X = de.fit_transform(df_month)
week = X.select(cs.starts_with("Datetime_day")).to_numpy()

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


def get_aspect(shape, H=2, W=20):
    cols, rows = shape
    aspect = (H / W) * (cols / rows)
    return aspect


titlefontsize = 38

fig, axs = plt.subplots(
    4,
    1,
    figsize=(20, 15),
    layout="constrained",
)

ax_hour = axs[0]
ax_week = axs[1]
ax_month = axs[2]
ax_year = axs[3]

# Plot for hours in a day
ax_hour.imshow(hour.T, aspect=get_aspect(hour.shape))
major_tick_positions = range(0, 48 * 2, 48)
minor_tick_positions = range(0, 48 * 2, 12)
ax_hour.set_xticks(major_tick_positions)
ax_hour.set_xticks(minor_tick_positions, minor=True)
ax_hour.xaxis.set_minor_formatter(ticker.FuncFormatter(hour_tick_formatter))
ax_hour.xaxis.set_major_formatter(ticker.FuncFormatter(half_hour_tick_formatter))
ax_hour.set_title("Hourly", fontsize=titlefontsize)
ax_hour.set_yticks([])

# Plot for weekdays in a week
ax_week.imshow(day[: 24 * 7].T, aspect=get_aspect(day.shape))
major_tick_positions = range(0, 24 * 7, 24)
minor_tick_positions = range(0, 24 * 7, 12)
ax_week.set_xticks(major_tick_positions)
ax_week.set_xticks(minor_tick_positions, minor=True)
ax_week.xaxis.set_major_formatter(ticker.FuncFormatter(day_tick_formatter))
ax_week.set_title("Daily", fontsize=titlefontsize)
ax_week.set_yticks([])

# Plot for days in a month
ax_month.imshow(week[:].T, aspect=get_aspect(week.shape))
major_tick_positions = range(0, 2 * 31, 7 * 2)
ax_month.set_xticks(major_tick_positions)
ax_month.xaxis.set_major_formatter(ticker.FuncFormatter(day_in_month_tick_formatter))
ax_month.set_title("Monthly", fontsize=titlefontsize)
ax_month.set_yticks([])

# Plot for months in a year
ax_year.imshow(month[:].T, aspect=get_aspect(month.shape))
first_days = []
base_time = datetime(2022, 1, 1)
for i in range((datetime(2023, 1, 1) - base_time).days):
    date = base_time + dt.timedelta(days=i)
    if date.day == 1:
        first_days.append(i)

ax_year.set_xticks(first_days)
plt.setp(ax_year.get_xticklabels(), rotation=45)
ax_year.xaxis.set_major_formatter(ticker.FuncFormatter(first_of_month_tick_formatter))
ax_year.set_title("Yearly", fontsize=titlefontsize)
ax_year.set_yticks([])

fig.savefig("periodic_features.png", dpi=300, bbox_inches="tight")

# Make backgrounds transparent and save light mode version (for dark websites)
fig.patch.set_alpha(0)
for ax in axs:
    ax.patch.set_alpha(0)
    ax.title.set_color("white")
    ax.tick_params(axis="both", which="both", colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    # Make spines lighter for dark mode
    for spine in ax.spines.values():
        spine.set_color("white")

# Save dark mode version (light text for dark backgrounds)
fig.savefig(
    "periodic_features_dark_transparent.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)

# Now save light mode version (dark text for light backgrounds)
for ax in axs:
    ax.title.set_color("black")
    ax.tick_params(axis="both", which="both", colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    # Make spines darker for light mode
    for spine in ax.spines.values():
        spine.set_color("black")

# Save light mode version (dark text for light backgrounds)
fig.savefig(
    "periodic_features_light_transparent.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)
# %%
