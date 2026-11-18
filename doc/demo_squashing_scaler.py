"""
This script is used to generate the plots that show the effect of the
SquashingScaler that are shown in the home page of the website.

It saves two transparent PNG files, following the same approach as
``demo_periodic_features.py``:

- ``squashing_scaler_light.png``: dark text, for light backgrounds
- ``squashing_scaler_dark.png``: light text, for dark backgrounds

The images are written to ``doc/_static`` so that they can be referenced
directly from the landing page template.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler

from skrub import SquashingScaler

STATIC_DIR = Path(__file__).parent / "_static"

# %%
# A column with a few extreme outliers, which is where the scaler helps most.
rng = np.random.default_rng(0)
x = np.linspace(0, 2 * np.pi, 50)
values = 30 * np.sin(x) + 10 * rng.normal(size=x.size)
extremes = np.array([50.0, -101.0, 1e12, 78.0, -40.0, 120.0, -1e9])
outlier_idx = [6, 12, 14, 18, 22, 32, 40]
values[outlier_idx] = extremes

df = pd.DataFrame({"feature": values})

# %%
scaler = SquashingScaler(max_absolute_value=5.0)
transformed = scaler.fit_transform(df)

# %%
plt.rcParams.update({"font.size": 30})

normal_color = "royalblue"
outlier_color = "orange"
inf_color = "crimson"
overlay_color = "olivedrab"

before = df["feature"].to_numpy(dtype=float)
after = transformed[:, 0]
ranks = np.arange(before.size)
is_outlier = np.zeros(before.size, dtype=bool)
is_outlier[outlier_idx] = True
# Values too large to be drawn on the "Before" panel.
is_large = np.abs(before) >= 1e3

fig, axs = plt.subplots(
    2, 1, figsize=(18, 12), sharex=False, sharey=False, layout="constrained"
)

ax_before, ax_after = axs

# Highlight the range preserved by the scaler. ``axhspan`` always covers the
# full width of the axes (its x bounds are axes fractions), so use
# ``fill_between`` to keep the band within the plotted samples.
x0, x1 = (10, 20)
in_zoom = (ranks >= x0) & (ranks <= x1)
# Only outliers inside the highlighted region get a marker; elsewhere they are
# drawn as ordinary samples.
marked = is_outlier & in_zoom
for ax in axs:
    ax.fill_between((x0, x1), -200, 200, color=normal_color, alpha=0.15, zorder=0)
    ax.axhline(-5, color="black", lw=1, ls="--", alpha=0.5)
    ax.axhline(5, color="black", lw=1, ls="--", alpha=0.5)

# The extreme values are far off-scale in the "Before" panel, so they fall
# outside the visible y range; the "After" panel shows where the scaler puts
# them.
ax_before.scatter(
    ranks[~marked],
    before[~marked],
    s=40,
    color=normal_color,
)
ax_before.scatter(
    ranks[marked & ~is_large],
    before[marked & ~is_large],
    s=85,
    marker="D",
    color=outlier_color,
    zorder=5,
)

ax_before.set_ylabel("Feature value", fontsize=36)

ax_after.scatter(
    ranks[~marked],
    after[~marked],
    s=40,
    color=normal_color,
)
ax_after.scatter(
    ranks[marked & ~is_large],
    after[marked & ~is_large],
    s=85,
    marker="D",
    color=outlier_color,
    zorder=5,
)
# The large values are squashed to the bounds.
ax_after.scatter(
    ranks[marked & is_large],
    after[marked & is_large],
    marker="X",
    s=85,
    color=inf_color,
    zorder=5,
)

ax_after.set_ylabel("Squashed value", fontsize=36)
for ax in axs:
    ax.tick_params(axis="x", labelbottom=False)

# Draw the values in index order; the extreme values are clipped so the line
# stays within the axes.
ax_before.plot(ranks, np.clip(before, -200, 2000), color="black", alpha=0.15)
ax_after.plot(ranks, np.clip(before, -200, 200), color="black", alpha=0.15)
ax_after.plot(ranks, after, color=normal_color, alpha=0.5, zorder=0)

# StandardScaler for comparison. Because the outliers inflate the standard
# deviation, the bulk of the data collapses around zero.
standard = StandardScaler().fit_transform(before.reshape(-1, 1)).ravel()
ax_after.plot(
    ranks,
    standard,
    color=overlay_color,
    lw=2,
    ls="--",
    marker="o",
    ms=6,
    zorder=1,
)

ax_before.set_ylim(-130, 130)

ax_after.set_xlim(x0 - 0.9, x1 + 0.9)
ax_after.set_ylim(-10, 10)

# Make the lower panel read as a zoom of the shaded region: outline that region
# in the top panel and join its bottom corners to the corners of the lower
# panel.
zoom_y0, zoom_y1 = ax_before.get_ylim()
ax_before.add_patch(
    Rectangle(
        (x0, zoom_y0),
        x1 - x0,
        zoom_y1 - zoom_y0,
        fill=False,
        edgecolor=normal_color,
        lw=3,
        zorder=6,
    )
)
zoom_x0, zoom_x1 = ax_after.get_xlim()
zoom_top = ax_after.get_ylim()[1]
for x_top, x_bottom in ((x0, zoom_x0), (x1, zoom_x1)):
    fig.add_artist(
        ConnectionPatch(
            xyA=(x_top, zoom_y0),
            coordsA="data",
            axesA=ax_before,
            xyB=(x_bottom, zoom_top),
            coordsB="data",
            axesB=ax_after,
            color=normal_color,
            lw=3,
            ls="--",
            alpha=0.9,
            clip_on=False,
            zorder=1,
        )
    )

# The extreme values are far outside the visible y range, so mark their
# positions at the edge of the "Before" panel and label them with their value.
# Only the ones inside the highlighted region get a marker, and the label is
# placed just outside that region so it stays readable on the plain background.
for position in ranks[is_large & in_zoom]:
    value = before[position]
    above = value > 0
    marker_y = 0.97 if above else 0.03
    mantissa, exponent = f"{value:.0e}".split("e")
    label = f"{mantissa}e{int(exponent)}"
    ax_before.annotate(
        label,
        xy=(position, marker_y),
        xycoords=("data", "axes fraction"),
        xytext=(x1 + 1.5 if above else x0 - 1.5, marker_y + (-0.12 if above else 0.12)),
        textcoords=("data", "axes fraction"),
        ha="left" if above else "right",
        va="center",
        fontsize=30,
        arrowprops={"arrowstyle": "->", "lw": 2, "shrinkB": 0},
    )
    ax_before.plot(
        position,
        marker_y,
        transform=ax_before.get_xaxis_transform(),
        marker="X",
        ms=17,
        ls="none",
        color=inf_color,
        zorder=5,
    )

# Label the two curves directly instead of using a legend. Each label sits in
# the empty space at the top or bottom of the panel and a thin leader in the
# curve's color points at a segment of the curve rather than at a single
# sample, so the label reads as naming the line.
zoom_ranks = ranks[in_zoom]
squashing_peak = zoom_ranks[np.argmax(after[zoom_ranks])]
squashing_segment = ranks[squashing_peak : squashing_peak + 2]
# The standard-scaled line is flat near zero for the bulk of the samples
# because the outlier inflates the standard deviation. Point at a segment
# inside the visible x range, close to the label, so the leader stays short.
standard_segment = ranks[(ranks >= x1 - 3) & (ranks <= x1) & ~is_outlier]
for label, segment, values, color, y_text in [
    ("SquashingScaler", squashing_segment, after, normal_color, 0.88),
    ("StandardScaler", standard_segment, standard, overlay_color, 0.12),
]:
    text = ax_after.annotate(
        label,
        xy=(segment.mean(), values[segment].mean()),
        xytext=(0.97, y_text),
        textcoords="axes fraction",
        ha="right",
        va="center",
        color=color,
        fontsize=30,
        arrowprops={
            "arrowstyle": "-",
            "color": color,
            "lw": 1.5,
            "alpha": 0.7,
            "shrinkA": 6,
            "shrinkB": 6,
        },
    )
    # Keep the color coding when the figure is restyled for dark backgrounds.
    text.set_gid("keep-color")

# Keep the x tick labels readable: fewer, evenly spaced ticks.
ax_before.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

# The two panels have tick labels of different widths, so align the y labels
# explicitly to put them on the same vertical line.
fig.align_ylabels(axs)

suptitle = fig.suptitle("Squashing Scaler", fontsize=48)
ax_before.set_title("Robust scaling and clipping of extreme values", fontsize=38)


# %%
def style_for_background(fig, color):
    """Make the figure transparent and recolor text and spines."""
    fig.patch.set_alpha(0)
    for text in fig.texts:
        text.set_color(color)
    for ax in fig.axes:
        ax.patch.set_alpha(0)
        ax.title.set_color(color)
        ax.xaxis.label.set_color(color)
        ax.yaxis.label.set_color(color)
        ax.tick_params(axis="both", which="both", colors=color)
        for spine in ax.spines.values():
            spine.set_color(color)
        # Annotation texts and their arrows
        for text in ax.texts:
            if text.get_gid() == "keep-color":
                continue
            text.set_color(color)
            arrow = getattr(text, "arrow_patch", None)
            if arrow is not None:
                arrow.set_edgecolor(color)
                arrow.set_facecolor(color)


# %%
# Light mode version (dark text for light backgrounds)
style_for_background(fig, "black")
fig.savefig(
    STATIC_DIR / "squashing_scaler_light.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)

# %%
# Dark mode version (light text for dark backgrounds)
style_for_background(fig, "white")
fig.savefig(
    STATIC_DIR / "squashing_scaler_dark.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)
# %%
