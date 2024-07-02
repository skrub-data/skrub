import base64
import builtins
from pathlib import Path
import json

import numpy as np


from skrub import _dataframe as sbd
from skrub._dispatch import dispatch


def read(file_path):
    try:
        from polars import read_csv, read_parquet
    except ImportError:
        from pandas import read_csv, read_parquet
    if file_path is not None:
        file_path = Path(file_path)
    suffix = file_path.suffix
    if suffix == ".parquet":
        return read_parquet(file_path)
    if suffix == ".csv":
        return read_csv(file_path)
    raise ValueError(f"Cannot process file extension: {suffix}")


def get_dtype_name(column):
    return sbd.dtype(column).__class__.__name__


@dispatch
def slice(obj, *start_stop):
    raise NotImplementedError()

@slice.specialize("pandas")
def _slice_pandas(obj, *start_stop):
    return obj.iloc[builtins.slice(*start_stop)]

@slice.specialize("polars")
def _slice_polars(obj, *start_stop):
    start, stop, _  = builtins.slice(*start_stop).indices(sbd.shape(obj)[0])
    return obj.slice(start, stop - start)

@dispatch
def sum(col):
    raise NotImplementedError()

@sum.specialize("pandas", argument_type="Column")
def _sum_pandas_col(col):
    return col.sum()

@sum.specialize("polars", argument_type="Column")
def _sum_polars_col(col):
    return col.sum()

@dispatch
def min(col):
    raise NotImplementedError()

@min.specialize("pandas", argument_type="Column")
def _min_pandas_col(col):
    return col.min()

@min.specialize("polars", argument_type="Column")
def _min_polars_col(col):
    return col.min()

@dispatch
def max(col):
    raise NotImplementedError()

@max.specialize("pandas", argument_type="Column")
def _max_pandas_col(col):
    return col.max()

@max.specialize("polars", argument_type="Column")
def _max_polars_col(col):
    return col.max()


@dispatch
def std(col):
    raise NotImplementedError()

@std.specialize("pandas", argument_type="Column")
def _std_pandas_col(col):
    return col.std()

@std.specialize("polars", argument_type="Column")
def _std_polars_col(col):
    return col.std()

@dispatch
def mean(col):
    raise NotImplementedError()

@mean.specialize("pandas", argument_type="Column")
def _mean_pandas_col(col):
    return col.mean()

@mean.specialize("polars", argument_type="Column")
def _mean_polars_col(col):
    return col.mean()



@dispatch
def to_dict(df):
    raise NotImplementedError()

@to_dict.specialize("pandas", argument_type="DataFrame")
def _to_dict_pandas(df):
    return df.to_dict(orient='list')

@to_dict.specialize("polars", argument_type="DataFrame")
def _to_dict_polars(df):
    return df.to_dict(as_series=False)


def first_row_dict(dataframe):
    first_row = slice(dataframe, 0, 1)
    return {col_name: col[0] for col_name, col in to_dict(first_row).items()}


def to_row_list(dataframe):
    columns = to_dict(dataframe)
    rows = []
    for row_idx in range(sbd.shape(dataframe)[0]):
        rows.append([col[row_idx] for col in columns.values()])
    return {"header": list(columns.keys()), "data": rows}


@dispatch
def value_counts(column):
    raise NotImplementedError()

@value_counts.specialize("pandas", argument_type="Column")
def _value_counts_pandas(column):
    return column.rename('value').value_counts().reset_index()

@value_counts.specialize("polars", argument_type="Column")
def _value_counts_polars(column):
    return column.rename('value').value_counts()


@dispatch
def sort(df, by, descending=False):
    raise NotImplementedError()

@sort.specialize("pandas", argument_type="DataFrame")
def _sort_pandas_dataframe(df, by, descending=False):
    return df.sort_values(by=by, ascending=not descending, ignore_index=True)

@sort.specialize("polars", argument_type="DataFrame")
def _sort_polars_dataframe(df, by, descending=False):
    return df.sort(by=by, descending=descending)


def top_k_value_counts(column, k):
    counts = value_counts(column)
    n_unique = sbd.shape(counts)[0]
    counts = sort(counts, by='count', descending=True)
    counts = slice(counts, k)
    return n_unique, dict(zip(*to_dict(counts).values()))


@dispatch
def quantile(column, q, interpolation='nearest'):
    raise NotImplementedError()

@quantile.specialize("pandas", argument_type="Column")
def _quantile_pandas_column(column, q, interpolation='nearest'):
    return column.quantile(q, interpolation=interpolation)

@quantile.specialize("polars", argument_type="Column")
def _quantile_polars_column(column, q, interpolation='nearest'):
    return column.quantile(q, interpolation=interpolation)


def quantiles(column):
    return {q: quantile(column, q) for q in [0.0, 0.25, 0.5, 0.75, 1.0]}



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
            if isinstance(value, np.int_):
                return int(value)
            if isinstance(value, np.float_):
                return float(value)
            raise
