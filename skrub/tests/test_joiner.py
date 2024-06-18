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
    false_joiner = Joiner(aux_table=aux_table, main_key="Countryy", aux_key="country")

    with pytest.raises(
        ValueError,
        match="do not exist in 'X'",
    ):
        false_joiner.fit(main_table)


def test_wrong_aux_key(main_table, aux_table):
    false_joiner2 = Joiner(aux_table=aux_table, main_key="Country", aux_key="bad")
    with pytest.raises(
        ValueError,
        match="do not exist in 'aux_table'",
    ):
        false_joiner2.fit(main_table)


def test_multiple_keys(df_module):
    df = df_module.make_dataframe(
        {"Co": ["France", "Italia", "Deutchland"], "Ca": ["Paris", "Roma", "Berlin"]}
    )
    df2 = df_module.make_dataframe(
        {"CO": ["France", "Italy", "Germany"], "CA": ["Paris", "Rome", "Berlin"]}
    )
    joiner_list = Joiner(
        aux_table=df2,
        aux_key=["CO", "CA"],
        main_key=["Co", "Ca"],
        add_match_info=False,
    )
    result = joiner_list.fit_transform(df)

    expected = ns.concat_horizontal(df, df2)
    df_module.assert_frame_equal(result, expected)

    joiner_list = Joiner(
        aux_table=df2, aux_key="CA", main_key="Ca", add_match_info=False
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
    assert ns.to_list(ns.col(join, "Country_capitals")) == [
        "France",
        "Italy",
        "Germany",
    ]


def test_bad_ref_dist():
    table = pd.DataFrame({"A": [1, 2]})
    joiner = Joiner(table, key="A", ref_dist="bad")
    with pytest.raises(ValueError, match="got 'bad'"):
        joiner.fit(table)


@pytest.mark.parametrize("max_dist", [np.inf, float("inf"), "inf", None])
def test_max_dist(max_dist):
    table = pd.DataFrame({"A": [1, 2]})
    joiner = Joiner(table, key="A", max_dist=max_dist, suffix="_").fit(table)
    assert joiner.max_dist_ == np.inf
