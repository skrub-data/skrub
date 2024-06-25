import datetime

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from skrub import Joiner
from skrub._dataframe import _common as ns


@pytest.fixture
def main_table(df_module):
    return df_module.make_dataframe(
        {
            "Country": [
                "France",
                "Germany",
                "Italy",
            ]
        }
    )


@pytest.fixture
def aux_table(df_module):
    return df_module.make_dataframe(
        {
            "country": ["Germany", "French Republic", "Italia"],
            "Population": [84_000_000, 68_000_000, 59_000_000],
        }
    )


def test_fit_transform(main_table, aux_table):
    joiner = Joiner(
        aux_table=aux_table,
        main_key="Country",
        aux_key="country",
        add_match_info=False,
    )

    joiner.fit(main_table)
    big_table = joiner.transform(main_table)
    assert ns.shape(big_table) == (ns.shape(main_table)[0], 3)
    assert_array_equal(
        ns.to_numpy(ns.col(big_table, "Population")),
        ns.to_numpy(ns.col(aux_table, "Population"))[[1, 0, 2]],
    )


def test_wrong_main_key(main_table, aux_table):
    wrong_joiner = Joiner(aux_table=aux_table, main_key="wrong_key", aux_key="country")
    with pytest.raises(ValueError, match="do not exist in 'X'"):
        wrong_joiner.fit(main_table)


def test_wrong_aux_key(main_table, aux_table):
    wrong_joiner_2 = Joiner(
        aux_table=aux_table, main_key="Country", aux_key="wrong_key"
    )
    with pytest.raises(ValueError, match="do not exist in 'aux_table'"):
        wrong_joiner_2.fit(main_table)


def test_multiple_keys(df_module):
    df = df_module.make_dataframe(
        {"Co": ["France", "Italia", "Deutchland"], "Ca": ["Paris", "Roma", "Berlin"]}
    )
    df2 = df_module.make_dataframe(
        {"CO": ["France", "Italy", "Germany"], "CA": ["Paris", "Rome", "Berlin"]}
    )
    joiner_list = Joiner(
        aux_table=df2,
        main_key=["Co", "Ca"],
        aux_key=["CO", "CA"],
        add_match_info=False,
    )
    result = joiner_list.fit_transform(df)

    expected = ns.concat_horizontal(df, df2)
    df_module.assert_frame_equal(result, expected)

    joiner_list = Joiner(
        aux_table=df2, main_key="Ca", aux_key="CA", add_match_info=False
    )
    result = joiner_list.fit_transform(df)
    df_module.assert_frame_equal(result, expected)


def test_pandas_aux_table_index():
    main_table = pd.DataFrame({"Country": ["France", "Italia", "Georgia"]})
    aux_table = pd.DataFrame(
        {
            "Country": ["Germany", "France", "Italy"],
            "Capital": ["Berlin", "Paris", "Rome"],
        }
    )
    aux_table.index = [2, 1, 0]

    joiner = Joiner(
        aux_table,
        key="Country",
        suffix="_capitals",
    )
    join = joiner.fit_transform(main_table)
    assert join["Country_capitals"].to_list() == [
        "France",
        "Italy",
        "Germany",
    ]


def test_wrong_ref_dist(main_table, aux_table):
    joiner = Joiner(
        aux_table, main_key="Country", aux_key="country", ref_dist="wrong_ref_dist"
    )
    with pytest.raises(
        ValueError, match=r"'ref_dist' should be one of.* Got 'wrong_ref_dist'"
    ):
        joiner.fit(main_table)


@pytest.mark.parametrize("max_dist", [np.inf, float("inf"), "inf", None])
def test_max_dist(main_table, aux_table, max_dist):
    joiner = Joiner(
        aux_table, main_key="Country", aux_key="country", max_dist=max_dist
    ).fit(main_table)
    assert joiner.max_dist_ == np.inf


def test_missing_values(df_module):
    df = df_module.make_dataframe({"A": [None, "hollywood", "beverly"]})
    joiner = Joiner(df, key="A", suffix="_")
    out = joiner.fit_transform(df)
    assert ns.shape(out) == (3, 5)


def test_fit_transform_numeric(df_module):
    df = df_module.make_dataframe({"A": [4.5, 0.5, 1, -1.5]})
    joiner = Joiner(df, key="A", suffix="_")
    out = joiner.fit_transform(df)
    assert ns.shape(out) == (4, 5)


def test_fit_transform_datetimes(df_module):
    values = [
        datetime.datetime.fromisoformat(dt)
        for dt in [
            "2020-02-03T12:30:05",
            "2021-03-15T00:37:15",
            "2022-02-13T17:03:25",
        ]
    ]
    df = df_module.make_dataframe({"A": values})
    joiner = Joiner(df, key="A", suffix="_")
    joiner.fit_transform(df)


def test_preprocessing(df_module):
    df1 = df_module.make_dataframe(dict(date=["2021-10-01"], v=["A"]))
    df2 = df_module.make_dataframe(
        dict(date=["2021-09-17", "2012-10-01"], v=["A", "B"])
    )
    out = Joiner(df2, key="date", suffix="_", add_match_info=False).fit_transform(df1)
    assert ns.to_list(ns.col(out, "v_")) == ["A"]
