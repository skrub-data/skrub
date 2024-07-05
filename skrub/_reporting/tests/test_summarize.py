import datetime

import numpy as np
import pytest
import zoneinfo
from sklearn.utils.fixes import parse_version

from skrub._reporting._summarize import summarize_dataframe


@pytest.mark.parametrize("order_by", [None, "date.utc"])
@pytest.mark.parametrize("with_plots", [False, True])
def test_summarize(df_module, air_quality, order_by, with_plots):
    if df_module.name == "polars":
        import polars as pl

        if parse_version(pl.__version__) <= parse_version("1.0.0") and parse_version(
            "2.0.0"
        ) <= parse_version(np.__version__):
            pytest.xfail("polars 1.0.0 does not support numpy 2 causing segfaults")
    summary = summarize_dataframe(
        air_quality, with_plots=with_plots, order_by=order_by, title="the title"
    )
    assert summary["title"] == "the title"
    assert summary["n_columns"] == 11
    assert summary["n_constant_columns"] == 4
    assert summary["n_rows"] == 207
    assert summary["head"]["header"] == [
        "city",
        "country",
        "date.utc",
        "location",
        "parameter",
        "value",
        "unit",
        "loc_with_nulls",
        "all_null",
        "constant_numeric",
        "constant_datetime",
    ]
    assert summary["tail"]["header"] == summary["head"]["header"]
    assert summary["head"]["data"][0] == [
        "Paris",
        "FR",
        datetime.datetime(2019, 6, 21, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC")),
        "FR04014",
        "no2",
        20.0,
        "µg/m³",
        None,
        None,
        2.7,
        datetime.datetime(2024, 7, 5, 12, 17, 29, 427865),
    ]
    assert len(summary["head"]["data"]) == len(summary["tail"]["data"]) == 5
    assert summary["first_row_dict"] == {
        "all_null": None,
        "city": "Paris",
        "country": "FR",
        "date.utc": datetime.datetime(
            2019, 6, 21, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC")
        ),
        "loc_with_nulls": None,
        "location": "FR04014",
        "parameter": "no2",
        "unit": "µg/m³",
        "value": 20.0,
        "constant_numeric": 2.7,
        "constant_datetime": datetime.datetime(2024, 7, 5, 12, 17, 29, 427865),
    }
    assert summary["dataframe"] is air_quality
    assert summary["dataframe_module"] == df_module.name

    # checking columns

    assert len(summary["columns"]) == summary["n_columns"]
    c = dict(summary["columns"][0])
    assert "string" in c["dtype"].lower() or "object" in c["dtype"].lower()
    c["dtype"] = "string"
    assert round(c["unique_proportion"], 3) == 0.014
    c["unique_proportion"] = 0.014
    if with_plots:
        assert c["value_counts_plot"].startswith("<?xml")
        assert c["plot_names"] == ["value_counts_plot"]
        c["plot_names"] = []
        c.pop("value_counts_plot")
    assert c == {
        "dtype": "string",
        "high_cardinality": False,
        "n_unique": 3,
        "name": "city",
        "null_count": 0,
        "null_proportion": 0.0,
        "nulls_level": "ok",
        "plot_names": [],
        "position": 0,
        "unique_proportion": 0.014,
        "value_counts": {"Antwerpen": 9, "London": 97, "Paris": 101},
        "value_is_constant": False,
    }

    assert summary["columns"][4]["constant_value"] == "no2"
    assert summary["columns"][4]["value_is_constant"]
    assert summary["columns"][5]["quantiles"] == {
        0.0: 3.0,
        0.25: 17.6,
        0.5: 25.0,
        0.75: 33.0 if df_module.name == "polars" else 32.8,
        1.0: 78.3,
    }
    assert summary["columns"][7]["null_count"] == 69
    assert summary["columns"][7]["nulls_level"] == "warning"
    assert summary["columns"][8]["null_count"] == 207
    assert summary["columns"][8]["nulls_level"] == "critical"

    # checking top associations

    assert len(summary["top_associations"]) == 11
    asso = [
        d | {"cramer_v": round(d["cramer_v"], 1)} for d in summary["top_associations"]
    ]
    for top_asso in asso[:3]:
        if top_asso == {
            "cramer_v": 1.0,
            "left_column": "city",
            "right_column": "country",
        }:
            break
    else:
        assert False
    assert asso[-1] == {
        "cramer_v": 0.2,
        "left_column": "value",
        "right_column": "loc_with_nulls",
    }


def test_no_title(pd_module):
    summary = summarize_dataframe(pd_module.example_dataframe)
    assert "title" not in summary


# def test_constant_numeric_column()
