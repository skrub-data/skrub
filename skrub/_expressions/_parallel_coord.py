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
        )
    )


def get_parallel_coord_data(cv_results, metadata, colorscale=DEFAULT_COLORSCALE):
    prepared_columns = [
        _prepare_column(cv_results[col_name], col_name in metadata["log_scale_columns"])
        for col_name in cv_results.columns
    ]
    prepared_columns = [
        _add_jitter(column) if column["label"] != "score" else column
        for column in prepared_columns
    ]
    return dict(
        line=dict(
            color=cv_results["mean_test_score"],
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=dict(text="score")),
        ),
        dimensions=prepared_columns,
        labelangle=45,
    )


def _add_jitter(column):
    vals = column["values"]
    min_val, max_val = np.min(vals), np.max(vals)
    eps = (max_val - min_val) / 200
    column["values"] = column["values"] + np.random.uniform(
        low=-eps, high=eps, size=vals.shape[0]
    )
    return column


def _prepare_column(col, is_log_scale):
    if pd.api.types.is_bool_dtype(col) or not pd.api.types.is_numeric_dtype(col):
        return _prepare_obj_column(col)
    if is_log_scale or col.isna().any():
        return _prepare_numeric_column(col, is_log_scale)
    label = {
        "mean_test_score": "score",
        "mean_fit_time": "fit time",
        "mean_score_time": "score time",
    }.get(col.name, col.name)
    return {"label": label, "values": col.to_numpy()}


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


def _prepare_numeric_column(col, log_scale):
    vals = col.to_numpy()
    if log_scale:
        vals = np.log(vals)
    min_val, max_val = np.nanmin(vals), np.nanmax(vals)
    tickvals = np.linspace(min_val, max_val, 10).tolist()
    if pd.api.types.is_integer_dtype(col):
        ticktext = [str(int(np.round(np.exp(v)))) for v in tickvals]
    else:
        ticktext = list(
            map("{:.2g}".format, np.exp(tickvals) if log_scale else tickvals)
        )
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
