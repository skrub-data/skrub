import base64
import json
import numbers
import re
import unicodedata

import numpy as np

from skrub import _dataframe as sbd
from skrub._dispatch import dispatch


def get_dtype_name(column):
    return sbd.dtype(column).__class__.__name__


@dispatch
def to_dict(df):
    raise NotImplementedError()


@to_dict.specialize("pandas", argument_type="DataFrame")
def _to_dict_pandas(df):
    return df.to_dict(orient="list")


@to_dict.specialize("polars", argument_type="DataFrame")
def _to_dict_polars(df):
    return df.to_dict(as_series=False)


def first_row_dict(dataframe):
    first_row = sbd.slice(dataframe, 0, 1)
    return {col_name: col[0] for col_name, col in to_dict(first_row).items()}


def top_k_value_counts(column, k):
    counts = sbd.value_counts(column)
    n_unique = sbd.shape(counts)[0]
    counts = sbd.sort(counts, by="count", descending=True)
    counts = sbd.slice(counts, k)
    return n_unique, list(zip(*to_dict(counts).values()))


def quantiles(column):
    return {q: sbd.quantile(column, q) for q in [0.0, 0.25, 0.5, 0.75, 1.0]}


def ellide_string(s, max_len=30):
    """Shorten a string so it can be used as a plot axis title or label."""
    if not isinstance(s, str):
        return s
    # normalize whitespace
    s = re.sub(r"\s+", " ", s)
    if len(s) <= max_len:
        return s
    shown_text = s[:max_len].strip()
    ellipsis = "…"
    end = ""

    # The ellipsis, like most punctuation, is a neutral character (it has no
    # direction). As here it is the last character in the sentence, its
    # direction will be that of the paragraph and it will probably be displayed
    # on the right: if we have truncated text in a right-to-left script the
    # ellipsis will be on the wrong side, at the beginning of the text. So if
    # the last character before truncation has a RTL direction, we insert after
    # the ellipsis a right-to-left mark (U200F), which is a zero-width space
    # with RTL direction. Thus the ellipsis is surrounded by 2 strong RTL
    # characters and it will be displayed RTL as well -- on the correct side of
    # the ellided text.
    if shown_text and unicodedata.bidirectional(shown_text[-1]) in [
        "R",
        "AL",
        "RLE",
        "RLO",
        "RLI",
    ]:
        end = "\u200f"
    return shown_text + ellipsis + end


def format_number(number):
    if isinstance(number, numbers.Integral):
        return f"{number:,}"
    if isinstance(number, numbers.Real):
        return f"{number:#.3g}"
    return str(number)


def format_percent(proportion):
    if 0.0 < proportion < 0.001:
        return "< 0.1%"
    return f"{proportion:0.1%}"


def svg_to_img_src(svg):
    encoded_svg = base64.b64encode(svg.encode("UTF-8")).decode("UTF-8")
    return f"data:image/svg+xml;base64,{encoded_svg}"


class JSONEncoder(json.JSONEncoder):
    def default(self, value):
        try:
            return super().default(value)
        except TypeError:
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            raise
