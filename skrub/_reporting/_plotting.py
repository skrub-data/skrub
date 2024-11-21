"""Generate the plots shown in the reports.

The figures are returned in the form of svg strings.
"""

import functools
import io
import re
import warnings

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from skrub import _dataframe as sbd

from . import _utils

__all__ = ["COLORS", "COLOR_0", "histogram", "line", "value_counts"]

# from matplotlib import colormaps, colors
# _TAB10 = list(map(colors.rgb2hex, colormaps.get_cmap("tab10").colors))


# sns.color_palette('muted').as_hex()
_SEABORN = [
    "#4878d0",
    "#ee854a",
    "#6acc64",
    "#d65f5f",
    "#956cb4",
    "#8c613c",
    "#dc7ec0",
    "#797979",
    "#d5bb67",
    "#82c6e2",
]

COLORS = _SEABORN
COLOR_0 = COLORS[0]


def _plot(plotting_fun):
    """Set the maptlotib config & silence some warnings for all report plots.

    All the plotting functions exposed by this module should be decorated with
    `_plot`.
    """

    @functools.wraps(plotting_fun)
    def plot_with_config(*args, **kwargs):
        # This causes matplotlib to insert labels etc as text in the svg rather
        # than drawing the glyphs.
        with matplotlib.rc_context({"svg.fonttype": "none"}):
            with warnings.catch_warnings():
                # We do not care about missing glyphs because the text is
                # rendered & the viewbox is recomputed in the browser.
                warnings.filterwarnings("ignore", "Glyph.*missing from font")
                warnings.filterwarnings(
                    "ignore", "Matplotlib currently does not support Arabic natively"
                )
                return plotting_fun(*args, **kwargs)

    return plot_with_config


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _to_em(pt_match):
    attr, pt = pt_match.groups()
    pt = float(pt)
    px = pt * 96 / 72
    em = px / 16
    return f'{attr}="{em:.2f}em"'


def _serialize(fig):
    buffer = io.BytesIO()
    fig.patch.set_visible(False)
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    out = buffer.getvalue().decode("UTF-8")
    out = re.sub(r'(width|height)="([0-9.]+)pt"', _to_em, out)
    plt.close(fig)
    return out


def _rotate_ticklabels(ax):
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment="right")


def _get_adjusted_fig_size(fig, ax, direction, target_size):
    size_display = getattr(ax.get_window_extent(), direction)
    size = fig.dpi_scale_trans.inverted().transform((size_display, 0))[0]
    dim = 0 if direction == "width" else 1
    fig_size = fig.get_size_inches()[dim]
    return target_size * (fig_size / size)


def _adjust_fig_size(fig, ax, target_w, target_h):
    """Rescale a figure to the target width and height.

    This allows us to have all figures of a given type (bar plots, lines or
    histograms) have the same size, so that the displayed report looks more
    uniform, without having to do manual adjustments to account for the length
    of labels, occurrence of titles etc. We let pyplot generate the figure
    without any size constraints then resize it and thus let matplotlib deal
    with resizing the appropriate elements (eg shorter bars when the labels
    take more horizontal space).
    """
    w = _get_adjusted_fig_size(fig, ax, "width", target_w)
    h = _get_adjusted_fig_size(fig, ax, "height", target_h)
    fig.set_size_inches((w, h))


@_plot
def histogram(col, color=COLOR_0):
    """Histogram for a numeric column."""
    col = sbd.drop_nulls(col)
    values = sbd.to_numpy(col)
    if np.issubdtype(values.dtype, np.floating):
        values = values[np.isfinite(values)]
    fig, ax = plt.subplots()
    _despine(ax)
    ax.hist(values, color=color)
    if sbd.is_any_date(col):
        _rotate_ticklabels(ax)
    _adjust_fig_size(fig, ax, 2.0, 1.0)
    return _serialize(fig)


@_plot
def line(x_col, y_col):
    """Line plot for a numeric column.

    ``x_col`` provides the x-axis values, ie the sorting column (corresponding
    to the report's ``order_by`` parameter). ``y_col`` is the column to plot as
    a function of x.
    """
    x = sbd.to_numpy(x_col)
    y = sbd.to_numpy(y_col)
    fig, ax = plt.subplots()
    _despine(ax)
    ax.plot(x, y)
    ax.set_xlabel(_utils.ellide_string(x_col.name))
    if sbd.is_any_date(x_col):
        _rotate_ticklabels(ax)
    _adjust_fig_size(fig, ax, 2.0, 1.0)
    return _serialize(fig)


@_plot
def value_counts(value_counts, n_unique, n_rows, color=COLOR_0):
    """Bar plot of the frequencies of the most frequent values in a column.

    Parameters
    ----------
    value_counts : list
        Pairs of (value, count). Must be sorted from most to least frequent.

    n_unique : int
        Cardinality of the plotted column, used to determine if all unique
        values are plotted or if there are too many and some have been
        ommitted. The figure's title is adjusted accordingly.

    n_rows : int
        Total length of the column, used to convert the counts to proportions.

    color : str
        The color for the bars.

    Returns
    -------
    str
        The plot as a XML string.
    """
    values = [_utils.ellide_string(v) for v, _ in value_counts][::-1]
    counts = [c for _, c in value_counts][::-1]
    if n_unique > len(value_counts):
        title = f"{len(value_counts)} most frequent"
    else:
        title = None
    fig, ax = plt.subplots()
    _despine(ax)
    rects = ax.barh(list(map(str, range(len(values)))), counts, color=color)
    percent = [_utils.format_percent(c / n_rows) for c in counts]
    large_percent = [
        f"{p: >6}" if c > counts[-1] / 2 else "" for (p, c) in zip(percent, counts)
    ]
    small_percent = [
        p if c <= counts[-1] / 2 else "" for (p, c) in zip(percent, counts)
    ]
    ax.bar_label(rects, large_percent, padding=-30, color="black", fontsize=8)
    ax.bar_label(rects, small_percent, padding=5, color="black", fontsize=8)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(list(map(str, values)))
    if title is not None:
        ax.set_title(title)

    _adjust_fig_size(fig, ax, 1.0, 0.2 * len(values))
    return _serialize(fig)
