import pathlib
import re
import secrets

import jinja2
import pandas as pd

try:
    from skrub import _selectors as s

    _SELECTORS_AVAILABLE = True
except ImportError:
    _SELECTORS_AVAILABLE = False

from skrub import _dataframe as sbd

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
    if not _SELECTORS_AVAILABLE:
        return _get_column_filters_no_selectors(df)
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
        if selector_name in _FILTER_NAMES:
            display_name = _FILTER_NAMES[selector_name]
        else:
            display_name = re.sub(r"^\((.*)\)$", r"\1", repr(selector))
            display_name = selector_name.replace("~", "NOT ").replace("()", "")
        filters[selector_name] = {
            "display_name": display_name,
            "columns": selector.expand(df),
        }
    return filters


def _get_column_filters_no_selectors(df):
    # temporary manual filtering until selectors PR is merged
    filters = {}
    if sbd.shape(df)[1] > 10:
        first_10 = sbd.column_names(df)[:10]
        filters["first_10"] = {"display_name": "First 10 columns", "columns": first_10}
    col_names = sbd.column_names(df)
    filters["all()"] = {"display_name": "All columns", "columns": col_names}

    def add_filt(f, name):
        filters[name] = {
            "display_name": name,
            "columns": [c for c in col_names if f(sbd.col(df, c))],
        }
        filters[f"~{name}"] = {
            "display_name": f"~{name}",
            "columns": [c for c in col_names if c not in filters[name]],
        }

    add_filt(sbd.is_numeric, "numeric()")
    add_filt(sbd.is_string, "string()")
    add_filt(sbd.is_categorical, "categorical()")
    add_filt(sbd.is_any_date, "any_date()")
    return filters


def to_html(summary, standalone=True, column_filters=None):
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
