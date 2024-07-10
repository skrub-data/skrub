"""Generate the HTML for TableReport."""
import pathlib
import secrets

import jinja2
import pandas as pd

from skrub import _dataframe as sbd
from skrub import _selectors as s

from . import _utils

_FILTER_NAMES = {
    "all()": "All columns",
    "has_nulls()": "Columns with null values",
    "(~has_nulls())": "Columns without null values",
    "numeric()": "Numeric columns",
    "(~numeric())": "Non-numeric columns",
    "string()": "String columns",
    "(~string())": "Non-string columns",
    "categorical()": "Categorical columns",
    "(~categorical())": "Non-categorical columns",
    "any_date()": "Datetime columns",
    "(~any_date())": "Non-datetime columns",
}


def _get_jinja_env():
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            pathlib.Path(__file__).resolve().parent / "_data" / "templates",
            encoding="UTF-8",
        ),
        autoescape=True,
    )
    for function_name in [
        "format_number",
        "format_percent",
        "svg_to_img_src",
        "filter_equal_snippet",
        "filter_isin_snippet",
    ]:
        env.filters[function_name] = getattr(_utils, function_name)
    env.filters["is_null"] = pd.isna
    return env


def _get_column_filters(df):
    filters = {}
    if sbd.shape(df)[1] > 10:
        filters["first_10"] = {
            "display_name": "First 10 columns",
            "columns": sbd.column_names(df)[:10],
        }
    all_selectors = [s.all()]
    for selector in [
        s.has_nulls(),
        s.numeric(),
        s.string(),
        s.categorical(),
        s.any_date(),
    ]:
        all_selectors.extend([selector, ~selector])
    for selector in all_selectors:
        selector_name = repr(selector)
        display_name = _FILTER_NAMES[selector_name]
        filters[selector_name] = {
            "display_name": display_name,
            "columns": selector.expand(df),
        }
    return filters


def to_html(summary, standalone=True, column_filters=None):
    """Given a dataframe summary, generate the HTML string.

    Parameters
    ----------
    summary : dict
        A dict containing the information about the dataframe, created by
        ``_summarize.summarize_dataframe``.
    standalone : bool, default=True
        Whether to generate a full HTML page (``standalone=True``), or only an
        HTML fragment which can be inserted into another page or the output of
        a jupyter notebook cell (``standalone=False``).
    column_filters : dict
        A dict for adding custom entries to the column filter dropdown menu.
        Each key is an id for the filter (e.g. ``"all()"``) and the value is a
        mapping with the keys ``display_name`` (the name shown in the menu,
        e.g. ``"All columns"``) and ``columns`` (a list of column names).

    Returns
    -------
    str
        The report as a string (containing HTML).
    """
    column_filters = column_filters if column_filters is not None else {}
    jinja_env = _get_jinja_env()
    if standalone:
        template = jinja_env.get_template("standalone-report.html")
    else:
        template = jinja_env.get_template("inline-report.html")
    default_filters = _get_column_filters(summary["dataframe"])
    return template.render(
        {
            "summary": summary,
            # prioritize user-provided filters and keep them at the beginning
            "column_filters": column_filters | {
                k: v for (k, v) in default_filters.items() if k not in column_filters
            },
            "report_id": f"report_{secrets.token_hex()[:8]}",
        }
    )
