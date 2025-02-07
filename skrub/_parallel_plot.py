import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

__all__ = ["get_parallel_coord_data", "plot_parallel_coord", "DEFAULT_COLORSCALE"]
DEFAULT_COLORSCALE = "bluered"


def plot_parallel_coord(cv_results, metadata, colorscale=DEFAULT_COLORSCALE):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("please install plotly.")
        return None
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
    return dict(
        line=dict(
            color=cv_results["mean_test_score"],
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=dict(text="score")),
        ),
        dimensions=prepared_columns,
    )


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
    encoder = OrdinalEncoder()
    encoded_col = encoder.fit_transform(pd.DataFrame({col.name: col})).ravel()
    return {
        "label": col.name,
        "values": encoded_col,
        "tickvals": np.arange(len(encoder.categories_[0])),
        "ticktext": encoder.categories_[0],
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
