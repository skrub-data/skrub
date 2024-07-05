import base64
import json

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


def to_row_list(dataframe):
    columns = to_dict(dataframe)
    rows = []
    for row_idx in range(sbd.shape(dataframe)[0]):
        rows.append([col[row_idx] for col in columns.values()])
    return {"header": list(columns.keys()), "data": rows}


def top_k_value_counts(column, k):
    counts = sbd.value_counts(column)
    n_unique = sbd.shape(counts)[0]
    counts = sbd.sort(counts, by="count", descending=True)
    counts = sbd.slice(counts, k)
    return n_unique, dict(zip(*to_dict(counts).values()))


def quantiles(column):
    return {q: sbd.quantile(column, q) for q in [0.0, 0.25, 0.5, 0.75, 1.0]}


def ellide_string(s, max_len=100):
    if not isinstance(s, str):
        return s
    if len(s) <= max_len:
        return s
    if max_len < 30:
        return s[:max_len] + "…"
    shown_len = max_len - 30
    truncated = len(s) - shown_len
    return s[:shown_len] + f"[…{truncated} more chars]"


def ellide_string_short(s):
    return ellide_string(s, 29)


def format_number(number):
    if not isinstance(number, float):
        return str(number)
    return f"{number:#.3g}"


def format_percent(proportion):
    if 0.0 < proportion < 0.001:
        return "< 0.1%"
    return f"{proportion:0.1%}"


def svg_to_img_src(svg):
    encoded_svg = base64.b64encode(svg.encode("UTF-8")).decode("UTF-8")
    return f"data:image/svg+xml;base64,{encoded_svg}"


def _pandas_filter_equal_snippet(value, column_name):
    if value is None:
        return f"df.loc[df[{column_name!r}].isnull()]"
    return f"df.loc[df[{column_name!r}] == {value!r}]"


def _pandas_filter_isin_snippet(values, column_name):
    return f"df.loc[df[{column_name!r}].isin({list(values)!r})]"


def _polars_filter_equal_snippet(value, column_name):
    if value is None:
        return f"df.filter(pl.col({column_name!r}).is_null())"
    return f"df.filter(pl.col({column_name!r}) == {value!r})"


def _polars_filter_isin_snippet(values, column_name):
    return f"df.filter(pl.col({column_name!r}).is_in({list(values)!r}))"


def filter_equal_snippet(value, column_name, dataframe_module="polars"):
    if dataframe_module == "polars":
        return _polars_filter_equal_snippet(value, column_name)
    if dataframe_module == "pandas":
        return _pandas_filter_equal_snippet(value, column_name)
    return f"Unknown dataframe library: {dataframe_module}"


def filter_isin_snippet(values, column_name, dataframe_module="polars"):
    if dataframe_module == "polars":
        return _polars_filter_isin_snippet(values, column_name)
    if dataframe_module == "pandas":
        return _pandas_filter_isin_snippet(values, column_name)
    return f"Unknown dataframe library: {dataframe_module}"


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
