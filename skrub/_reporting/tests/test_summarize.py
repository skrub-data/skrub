import datetime

import pytest
import zoneinfo

from skrub import _dataframe as sbd
from skrub._reporting import _interactions
from skrub._reporting._summarize import summarize_dataframe


@pytest.mark.parametrize("order_by", [None, "date.utc", "value"])
@pytest.mark.parametrize("with_plots", [False, True])
def test_summarize(monkeypatch, df_module, air_quality, order_by, with_plots):
    monkeypatch.setattr(_interactions, "_CATEGORICAL_THRESHOLD", 10)
    summary = summarize_dataframe(
        air_quality, with_plots=with_plots, order_by=order_by, title="the title"
    )
    assert summary["title"] == "the title"
    assert summary["n_columns"] == 11
    assert summary["n_constant_columns"] == 4
    assert summary["n_rows"] == 17
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
        "London",
        "GB",
        datetime.datetime(2019, 6, 13, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC")),
        "London Westminster",
        "no2",
        29.0,
        "µg/m³",
        None,
        None,
        2.7,
        datetime.datetime(2024, 7, 5, 12, 17, 29, 427865),
    ]
    assert len(summary["head"]["data"]) == len(summary["tail"]["data"]) == 5
    assert summary["first_row_dict"] == {
        "all_null": None,
        "city": "London",
        "constant_datetime": datetime.datetime(2024, 7, 5, 12, 17, 29, 427865),
        "constant_numeric": 2.7,
        "country": "GB",
        "date.utc": datetime.datetime(
            2019, 6, 13, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC")
        ),
        "loc_with_nulls": None,
        "location": "London Westminster",
        "parameter": "no2",
        "unit": "µg/m³",
        "value": 29.0,
    }
    assert summary["dataframe"] is air_quality
    assert summary["dataframe_module"] == df_module.name

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
        "dtype": "string",
        "high_cardinality": False,
        "n_unique": 2,
        "name": "city",
        "null_count": 0,
        "null_proportion": 0.0,
        "nulls_level": "ok",
        "plot_names": [],
        "position": 0,
        "unique_proportion": 0.118,
        "value_counts": {"London": 8, "Paris": 9},
        "value_is_constant": False,
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

    assert len(summary["top_associations"]) == 15
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
        "cramer_v": 0.5,
        "right_column": "loc_with_nulls",
        "left_column": "value",
    }


def test_no_title(pd_module):
    summary = summarize_dataframe(pd_module.example_dataframe, with_plots=False)
    assert "title" not in summary


def test_high_cardinality_column(pd_module):
    df = pd_module.make_dataframe({"s": [f"value {i}" for i in range(30)]})
    summary = summarize_dataframe(df, with_plots=True)
    assert "10 most frequent" in summary["columns"][0]["value_counts_plot"]


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
