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
    s = str(s)

    # normalize whitespace
    s = re.sub(r"\s+", " ", s)
    if len(s) <= max_len:
        return s
    shown_text = s[:max_len].strip()
    ellipsis = "…"
    end = ""

    # The ellipsis, like most punctuation, is a neutral character (it has no
    # writing direction). As here it is the last character in the sentence, its
    # direction will be that of the paragraph and it might be displayed on the
    # wrong side of the text (eg on the right, at the beginning of the text
    # rather than the end, if the text is written in a right-to-left script).
    # As a simple heuristic to correct this, we force the ellipsis to have the
    # same direction as the last character before the truncation. This is done
    # by appending a mark (a zero-width space with the writing direction we
    # want, so that the ellipsis is enclosed between 2 strong characters with
    # the same direction and thus inherits that direction).

    if shown_text:
        direction = unicodedata.bidirectional(shown_text[-1])
        if direction in [
            "R",
            "RLE",
            "RLO",
            "RLI",
        ]:
            # RIGHT-TO-LEFT MARK
            end = "\u200f"
        elif direction in ["AL"]:
            # ARABIC LETTER MARK
            end = "\u061c"
        elif direction in ["L", "LRE", "LRO", "LRI"]:
            # LEFT-TO-RIGHT MARK
            end = "\u200e"
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


def duration_to_numeric(col):
    seconds = sbd.total_seconds(col)
    q = sbd.quantile(sbd.abs(seconds), 0.9)
    HOUR = 3600
    DAY = HOUR * 24
    YEAR = DAY * 365.2425
    if q < 1e-3:
        return seconds * 1e6, "microsecond"
    if q < 1.0:
        return seconds * 1e3, "millisecond"
    if q < HOUR:
        return seconds, "second"
    if q < DAY:
        return seconds / HOUR, "hour"
    if q < YEAR:
        return seconds / DAY, "day"
    return seconds / YEAR, "year"


def strip_xml_declaration(svg):
    svg = re.sub(r"<\?xml.*?\?>", "", svg, flags=re.DOTALL)
    svg = re.sub(r"<!DOCTYPE.*?>", "", svg, flags=re.DOTALL)
    return svg
