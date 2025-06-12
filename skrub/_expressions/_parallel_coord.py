import textwrap

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

__all__ = ["get_parallel_coord_data", "plot_parallel_coord", "DEFAULT_COLORSCALE"]
DEFAULT_COLORSCALE = "bluered"


def plot_parallel_coord(cv_results, metadata, colorscale=DEFAULT_COLORSCALE):
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Please install plotly to display parallel coordinate plots.")

    return go.Figure(
        data=go.Parcoords(
            **get_parallel_coord_data(
                cv_results,
                metadata,
                colorscale=colorscale,
            )
        ),
        layout=go.Layout(font=dict(size=18)),
    )


def get_parallel_coord_data(cv_results, metadata, colorscale=DEFAULT_COLORSCALE):
    prepared_columns = [
        _prepare_column(
            cv_results[col_name],
            is_log_scale=col_name in metadata["log_scale_columns"],
            is_int=col_name in metadata["int_columns"],
        )
        for col_name in cv_results.columns
    ]
    prepared_columns = [
        _add_jitter(column) if column["label"] != "score" else column
        for column in prepared_columns
    ]
    return dict(
        line=dict(
            color=cv_results[metadata["col_score"]],
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=dict(text="score")),
        ),
        dimensions=prepared_columns,
        labelangle=15,
        labelside="top",
    )


def _add_jitter(column):
    vals = column["values"]
    min_val, max_val = np.min(vals), np.max(vals)
    argmin, argmax = np.argmin(vals), np.argmax(vals)
    eps = (max_val - min_val) / 200
    vals = column["values"] + np.random.uniform(low=-eps, high=eps, size=vals.shape[0])
    # plotly adds extra labels for the min and max of the range. So we make
    # sure we don't exceed the current bounds and that they are still attained
    # at least once to avoid having incorrect bounds displayed.
    vals = np.clip(vals, min_val, max_val)
    vals[argmin] = min_val
    vals[argmax] = max_val

    column["values"] = vals
    return column


def _wrap_label(column):
    """Add "<br>" to limit the width of a label to at most 20 chars"""
    label = "<br>\n".join(textwrap.wrap(column["label"], width=20))
    return {**column, "label": label}


def _prepare_column(col, *, is_log_scale, is_int):
    if pd.api.types.is_bool_dtype(col) or not pd.api.types.is_numeric_dtype(col):
        return _wrap_label(_prepare_obj_column(col))
    result = _prepare_numeric_column(col, is_log_scale=is_log_scale, is_int=is_int)
    result["label"] = {
        "mean_test_score": "score",
        "mean_fit_time": "fit time",
        "mean_score_time": "score time",
    }.get(result["label"], result["label"])
    return _wrap_label(result)


def _prepare_obj_column(col):
    is_null = col.isna().values
    encoder = OrdinalEncoder()
    encoded_not_null = encoder.fit_transform(
        col.dropna().astype(str).values[:, None]
    ).ravel()
    encoded = np.full(col.shape, -1.0)
    encoded[~is_null] = encoded_not_null
    categories = encoder.categories_[0]
    vals = list(range(len(categories)))
    if is_null.any():
        categories = ["Null"] + list(categories)
        vals = [-1.0, *vals]
    return {
        "label": col.name,
        "values": encoded,
        "tickvals": vals,
        "ticktext": categories,
    }


def _pick_format(vals):
    delta = (vals.max() - vals.min()) / (len(vals) + 1)
    if delta == 0.0 or any("e" in f"{v:g}" for v in vals):
        # only one values, or scientific notation -- bail for simplicity
        return "{:g}"
    # guess the necessary number of digits
    n = max(0, -int(np.floor(np.log10(delta))))
    return f"{{:.{n}f}}"


def _prepare_numeric_column(col, *, is_log_scale, is_int):
    vals = col.to_numpy()
    if is_log_scale:
        vals = np.log(vals)
    min_val, max_val = np.nanmin(vals), np.nanmax(vals)
    tickvals = np.unique(np.linspace(min_val, max_val, 10))
    if pd.api.types.is_integer_dtype(col) or is_int:
        if is_log_scale:
            tickvals = np.exp(tickvals)
        tickvals = np.unique(np.round(tickvals).astype(np.int64))
        tickvals_label_space = tickvals.tolist()
        if is_log_scale:
            tickvals = np.log(tickvals)
        tickvals = tickvals.tolist()
        ticktext = [str(val) for val in tickvals_label_space]
    else:
        tickvals_label_space = np.exp(tickvals) if is_log_scale else tickvals
        tickvals = tickvals.tolist()
        fmt = _pick_format(tickvals_label_space)
        ticktext = [fmt.format(val) for val in tickvals_label_space]
    if np.isnan(vals).any():
        tickvals = [min_val - (max_val - min_val) / 10] + tickvals
        ticktext = ["NaN"] + ticktext
        vals = np.where(~np.isnan(vals), vals, tickvals[0])
    return {
        "label": col.name,
        "values": vals,
        "tickvals": tickvals,
        "ticktext": ticktext,
    }
