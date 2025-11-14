import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from skrub import _column_associations
from skrub import _dataframe as sbd
from skrub._reporting import _sample_table
from skrub._reporting._summarize import summarize_dataframe
from skrub.conftest import skip_polars_installed_without_pyarrow


@pytest.mark.parametrize("order_by", [None, "date.utc", "value"])
@pytest.mark.parametrize("with_plots", [False, True])
@pytest.mark.parametrize("with_associations", [False, True])
@skip_polars_installed_without_pyarrow
def test_summarize(
    monkeypatch, df_module, air_quality, order_by, with_plots, with_associations
):
    monkeypatch.setattr(_column_associations, "_CATEGORICAL_THRESHOLD", 10)
    summary = summarize_dataframe(
        air_quality,
        with_plots=with_plots,
        with_associations=with_associations,
        order_by=order_by,
        title="the title",
    )
    assert summary["title"] == "the title"
    assert not summary["dataframe_is_empty"]
    assert summary["n_columns"] == 11
    assert summary["n_constant_columns"] == 4
    assert summary["n_rows"] == 17
    assert summary["dataframe"] is air_quality
    assert summary["dataframe_module"] == df_module.name
    assert summary["sample_table"]["start_i"] == (
        -2 if df_module.name == "pandas" else -1
    )
    assert summary["sample_table"]["stop_i"] == 10
    assert summary["sample_table"]["start_j"] == (
        -1 if df_module.name == "pandas" else 0
    )
    assert summary["sample_table"]["stop_j"] == 11

    # checking columns

    assert len(summary["columns"]) == summary["n_columns"]
    c = dict(summary["columns"][0])
    assert "string" in c["dtype"].lower() or "object" in c["dtype"].lower()
    c["dtype"] = "string"
    assert round(c["unique_proportion"], 3) == 0.118
    c["unique_proportion"] = 0.118
    if with_plots:
        assert c["value_counts_plot"].startswith("<?xml")
        assert c["plot_names"] == ["value_counts_plot"]
        c["plot_names"] = []
        c.pop("value_counts_plot")
    assert c == {
        "idx": 0,
        "dtype": "string",
        "is_ordered": False,
        "n_unique": 2,
        "name": "city",
        "null_count": 0,
        "null_proportion": 0.0,
        "nulls_level": "ok",
        "plot_names": [],
        "position": 0,
        "unique_proportion": 0.118,
        "value_counts": [("Paris", 9), ("London", 8)],
        "most_frequent_values": ["Paris", "London"],
        "value_is_constant": False,
        "is_duration": False,
        "is_high_cardinality": False,
    }

    assert summary["columns"][4]["constant_value"] == "no2"
    assert summary["columns"][4]["value_is_constant"]
    assert summary["columns"][5]["quantiles"] == {
        0.0: 5.0,
        0.25: 17.3,
        0.5: 27.0,
        0.75: 33.6,
        1.0: 78.3,
    }
    assert summary["columns"][7]["null_count"] == 9
    assert summary["columns"][7]["nulls_level"] == "warning"
    assert summary["columns"][8]["null_count"] == 17
    assert summary["columns"][8]["nulls_level"] == "critical"

    # checking top associations
    if with_associations:
        assert len(summary["top_associations"]) == 20
        asso = [
            d | {"cramer_v": round(d["cramer_v"], 1)}
            for d in summary["top_associations"]
        ]
        assert {
            tuple(sorted((a["left_column_name"], a["right_column_name"])))
            for a in asso[:3]
        } == {("city", "country"), ("city", "location"), ("country", "location")}
        assert asso[-1]["cramer_v"] == 0.0
    else:
        assert "top_associations" not in summary.keys()


def test_no_title(pd_module):
    summary = summarize_dataframe(pd_module.example_dataframe, with_plots=False)
    assert "title" not in summary


def test_high_cardinality_column(pd_module):
    df = pd_module.make_dataframe({"s": [f"value {i}" for i in range(30)]})
    summary = summarize_dataframe(df, with_plots=True)
    assert "10 most frequent" in summary["columns"][0]["value_counts_plot"]


@skip_polars_installed_without_pyarrow
def test_all_null(df_module):
    df = df_module.make_dataframe(
        {
            "a": sbd.to_float32(df_module.make_column("a", [None, None])),
            "b": sbd.to_datetime(
                df_module.make_column("b", ["", ""]), format="%Y-%m-%d", strict=False
            ),
            "c": sbd.to_string(df_module.make_column("c", [None, None])),
        }
    )
    summary = summarize_dataframe(df, with_plots=True)
    for col in summary["columns"]:
        assert col["null_proportion"] == 1.0


def test_empty_df(df_module):
    assert summarize_dataframe(df_module.empty_dataframe)["dataframe_is_empty"]


@pytest.fixture
def small_df_summary(df_module):
    def make_summary(df_size, **summarize_kwargs):
        df = df_module.make_dataframe(dict(a=[10.5] * df_size))
        return summarize_dataframe(df, **summarize_kwargs)

    return make_summary


@skip_polars_installed_without_pyarrow
def test_small_df(small_df_summary):
    summary = small_df_summary(11)
    thead, first_slice, ellipsis, last_slice = summary["sample_table"]["parts"]
    assert len(first_slice["rows"]) == 5
    assert len(last_slice["rows"]) == 5

    summary = small_df_summary(11, max_top_slice_size=2, max_bottom_slice_size=3)
    thead, first_slice, ellipsis, last_slice = summary["sample_table"]["parts"]
    assert len(first_slice["rows"]) == 2
    assert len(last_slice["rows"]) == 3

    summary = small_df_summary(10, max_top_slice_size=5, max_bottom_slice_size=0)
    thead, first_slice, ellipsis = summary["sample_table"]["parts"]
    assert len(first_slice["rows"]) == 5

    summary = small_df_summary(10)
    thead, first_slice = summary["sample_table"]["parts"]
    assert len(first_slice["rows"]) == 10

    summary = small_df_summary(9)
    thead, first_slice = summary["sample_table"]["parts"]
    assert len(first_slice["rows"]) == 9

    summary = small_df_summary(1)
    thead, first_slice = summary["sample_table"]["parts"]
    assert len(first_slice["rows"]) == 1

    summary = small_df_summary(0)
    thead, first_slice = summary["sample_table"]["parts"]
    assert len(first_slice["rows"]) == 0


def get_pivoted_df():
    # Example from https://pandas.pydata.org/docs/user_guide/reshaping.html#pivot-table

    df = pd.DataFrame(
        {
            "A": ["one", "one", "two", "three"] * 6,
            "B": ["A", "B", "C"] * 8,
            "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
            "D": np.random.randn(24),
            "E": np.random.randn(24),
            "F": [datetime.datetime(2013, i, 1) for i in range(1, 13)] + [
                datetime.datetime(2013, i, 15) for i in range(1, 13)
            ],
        }
    )

    df = pd.pivot_table(
        df,
        values="E",
        index=["B", "C"],
        columns=["A"],
        aggfunc=["sum", "mean"],
    )
    return df


def test_multi_index():
    df = get_pivoted_df()
    summary = summarize_dataframe(df)
    th = summary["sample_table"]["parts"][0]["rows"][0][0]
    assert th["colspan"] == 2
    assert th["rowspan"] == 1
    assert th["value"] is None

    df = get_pivoted_df()
    df.columns.names = [None, None]
    summary = summarize_dataframe(df)
    th = summary["sample_table"]["parts"][0]["rows"][0][0]
    assert th.get("colspan", 1) == 1
    assert th["rowspan"] == 3
    assert th["value"] == "B"

    df = get_pivoted_df()
    df.index.names = [None, None]
    summary = summarize_dataframe(df)
    th = summary["sample_table"]["parts"][0]["rows"][0][0]
    assert th["colspan"] == 2
    assert th.get("rowspan", 1) == 1
    assert th["value"] is None

    df = get_pivoted_df()
    df.index.names = [None, None]
    df.columns.names = [None, None]
    summary = summarize_dataframe(df)
    th = summary["sample_table"]["parts"][0]["rows"][0][0]
    assert th["colspan"] == 2
    assert th["rowspan"] == 3
    assert th["value"] is None


def test_level_names():
    # fallbacks in case some pandas non-multi index does not have the ".names" alias
    idx = SimpleNamespace()
    assert _sample_table._level_names(idx) == [None]
    idx.name = "the name"
    assert _sample_table._level_names(idx) == ["the name"]
    idx.names = ["a", "b"]
    assert _sample_table._level_names(idx) == ["a", "b"]


def test_duplicate_columns(pd_module):
    df = pd_module.make_dataframe({"a": [1, 2], "b": [3, 4]})
    df.columns = ["a", "a"]
    summary = summarize_dataframe(df)
    cols = summary["columns"]
    assert len(cols) == 2
    assert cols[0]["name"] == "a"
    assert cols[0]["mean"] == 1.5
    assert cols[1]["name"] == "a"
    assert cols[1]["mean"] == 3.5


@skip_polars_installed_without_pyarrow
def test_high_cardinality_columns(df_module):
    df = df_module.make_dataframe(
        {
            "low-cardinality": [0] * 100,
            "high-cardinality": range(100),
        }
    )
    summary = summarize_dataframe(df)
    cols = summary["columns"]
    assert not cols[0]["is_high_cardinality"]
    assert cols[1]["is_high_cardinality"]


@skip_polars_installed_without_pyarrow
def test_bool_column_mean(df_module):
    df = df_module.make_dataframe({"a": [True, False, True, True, False, True]})
    summary = summarize_dataframe(df)
    cols = summary["columns"]
    assert "mean" in cols[0]
