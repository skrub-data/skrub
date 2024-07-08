import io

from matplotlib import pyplot as plt

from skrub import _dataframe as sbd

from . import _utils

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


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _serialize(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    out = buffer.getvalue().decode("UTF-8")
    plt.close(fig)
    return out


def _rotate_ticklabels(ax):
    if len(ax.get_xticklabels()[0].get_text()) > 5:
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")


def _get_adjusted_fig_size(fig, ax, direction, target_size):
    size_display = getattr(ax.get_window_extent(), direction)
    size = fig.dpi_scale_trans.inverted().transform((size_display, 0))[0]
    dim = 0 if direction == "width" else 1
    fig_size = fig.get_size_inches()[dim]
    return target_size * (fig_size / size)


def _adjust_fig_size(fig, ax, target_w, target_h):
    w = _get_adjusted_fig_size(fig, ax, "width", target_w)
    h = _get_adjusted_fig_size(fig, ax, "height", target_h)
    fig.set_size_inches((w, h))


def histogram(col, color=COLOR_0):
    values = sbd.to_numpy(col)
    fig, ax = plt.subplots()
    _despine(ax)
    ax.hist(values, color=color)
    _rotate_ticklabels(ax)
    _adjust_fig_size(fig, ax, 2.0, 1.0)
    return _serialize(fig)


def line(x_col, y_col):
    x = sbd.to_numpy(x_col)
    y = sbd.to_numpy(y_col)
    fig, ax = plt.subplots()
    _despine(ax)
    ax.plot(x, y)
    ax.set_xlabel(_utils.ellide_string_short(x_col.name))
    _rotate_ticklabels(ax)
    _adjust_fig_size(fig, ax, 2.0, 1.0)
    return _serialize(fig)


def value_counts(value_counts, n_unique, n_rows, color=COLOR_0):
    values = [_utils.ellide_string_short(s) for s in value_counts.keys()][::-1]
    counts = list(value_counts.values())[::-1]
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
