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
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler

from skrub import SquashingScaler

STATIC_DIR = Path(__file__).parent / "_static"

# %%
# A column with a few extreme outliers, which is where the scaler helps most.
rng = np.random.default_rng(0)
values = rng.normal(loc=0.0, scale=1.0, size=50)
extremes = np.array([50.0, -101.0, 25.0, -40.0, 120.0, np.inf, -np.inf])
outlier_idx = rng.choice(len(values), size=len(extremes), replace=False)
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
finite = np.isfinite(before)
ranks = np.arange(before.size)
is_outlier = np.zeros(before.size, dtype=bool)
is_outlier[outlier_idx] = True

fig, axs = plt.subplots(
    2, 1, figsize=(18, 12), sharex=True, sharey=False, layout="constrained"
)

ax_before, ax_after = axs

# Highlight the range preserved by the scaler.
for ax in axs:
    ax.axhspan(-10, 10, color=normal_color, alpha=0.05)
    ax.axhline(-5, color="black", lw=1, ls="--", alpha=0.5)
    ax.axhline(5, color="black", lw=1, ls="--", alpha=0.5)

# Infinite values cannot be drawn, so only the finite ones are plotted here.
ax_before.scatter(
    ranks[finite & ~is_outlier],
    before[finite & ~is_outlier],
    s=40,
    color=normal_color,
)
ax_before.scatter(
    ranks[finite & is_outlier],
    before[finite & is_outlier],
    s=85,
    marker="D",
    color=outlier_color,
    zorder=5,
)

ax_before.set_ylabel("Feature value", fontsize=36)

ax_after.scatter(
    ranks[~is_outlier & finite],
    after[~is_outlier & finite],
    s=40,
    color=normal_color,
)
ax_after.scatter(
    ranks[is_outlier & finite],
    after[is_outlier & finite],
    s=85,
    marker="D",
    color=outlier_color,
    zorder=5,
)
# The infinite values are squashed to the bounds; mark them like in the
# "Before" panel.
ax_after.scatter(
    ranks[~finite],
    after[~finite],
    marker="X",
    s=85,
    color=inf_color,
    zorder=5,
)

ax_after.set_xlabel("Sample index", fontsize=36)
ax_after.set_ylabel("Squashed value", fontsize=36)

# Draw the values in index order; the infinities are clipped so the line
# stays within the axes.
ax_before.plot(ranks, np.clip(before, -200, 200), color="black", alpha=0.15)
ax_after.plot(ranks, np.clip(before, -200, 200), color="black", alpha=0.15)
ax_after.plot(ranks, after, color=normal_color, alpha=0.5, zorder=0)

# StandardScaler for comparison. It cannot handle the infinities, so it is
# fitted on the finite values only. Because the outliers inflate the standard
# deviation, the bulk of the data collapses around zero.
standard = np.full(before.shape, np.nan)
standard[finite] = StandardScaler().fit_transform(before[finite].reshape(-1, 1)).ravel()
ax_after.plot(
    ranks[finite],
    standard[finite],
    color=overlay_color,
    lw=2,
    ls="--",
    zorder=1,
)

ax_before.set_ylim(-150, 150)
ax_after.set_ylim(-20, 20)

# Label the two curves directly instead of using a legend. Each label sits in
# the empty space at the top or bottom of the panel and a thin leader in the
# curve's color points at a segment of the curve rather than at a single
# sample, so the label reads as naming the line: the steep segment next to the
# highest squashed value and the flat segment that runs across the positive
# infinity.
finite_ranks = ranks[finite]
right_half = finite_ranks[len(finite_ranks) // 2 :]
squashing_peak = right_half[np.argmax(after[right_half])]
squashing_segment = finite_ranks[
    np.searchsorted(finite_ranks, squashing_peak) : np.searchsorted(
        finite_ranks, squashing_peak
    )
    + 2
]
# The standard-scaled line skips the infinite values, so the segment crossing
# the positive infinity joins the samples on either side of it.
inf_rank = ranks[~finite][before[~finite] > 0][0]
standard_segment = finite_ranks[
    np.searchsorted(finite_ranks, inf_rank) - 1 : np.searchsorted(
        finite_ranks, inf_rank
    )
    + 1
]
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

# Point to the positions of the infinite values instead of plotting them.
# The label sits to the side of the marker and the arrow points at it
# diagonally: ``+inf`` is approached from the lower right, ``-inf`` from the
# upper left. A cross near the edge of the axes marks the off-scale value,
# leaving a gap with the arrow tip.
for position in ranks[~finite]:
    above = before[position] > 0
    side = 1 if above else -1
    marker_y = 0.97 if above else 0.03
    ax_before.annotate(
        "+inf" if above else "-inf",
        xy=(position + side * 0.8, marker_y - side * 0.02),
        xycoords=("data", "axes fraction"),
        xytext=(position + side * 7, marker_y - side * 0.14),
        textcoords=("data", "axes fraction"),
        ha="center",
        va="center",
        arrowprops={"arrowstyle": "->", "lw": 1.5, "shrinkB": 0},
    )
    ax_before.plot(
        position,
        0.97 if above else 0.03,
        transform=ax_before.get_xaxis_transform(),
        marker="X",
        ms=17,
        ls="none",
        color=inf_color,
        zorder=5,
    )

# Keep the x tick labels readable: fewer, evenly spaced ticks. Because the
# axes share x, setting the locator once is enough.
ax_before.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

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
