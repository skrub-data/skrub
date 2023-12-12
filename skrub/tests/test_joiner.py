import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from skrub import Joiner
from skrub._dataframe._polars import POLARS_SETUP

MODULES = [pd]
ASSERT_TUPLES = [(pd, assert_frame_equal)]

if POLARS_SETUP:
    import polars as pl
    from polars.testing import assert_frame_equal as assert_frame_equal_pl

    MODULES.append(pl)
    ASSERT_TUPLES.append((pl, assert_frame_equal_pl))


@pytest.mark.parametrize("px", MODULES)
def test_joiner(px):
    main_table = px.DataFrame(
        {
            "Country": [
                "France",
                "Germany",
                "Italy",
            ]
        }
    )

    aux_table = px.DataFrame(
        {
            "country": ["Germany", "French Republic", "Italia"],
            "Population": [84_000_000, 68_000_000, 59_000_000],
        }
    )
    joiner = Joiner(
        aux_table=aux_table,
        main_key="Country",
        aux_key="country",
        add_match_info=False,
    )

    joiner.fit(main_table)
    big_table = joiner.transform(main_table)
    assert big_table.shape == (main_table.shape[0], 3)
    assert_array_equal(
        big_table["Population"].to_numpy(),
        aux_table["Population"].to_numpy()[[1, 0, 2]],
    )

    false_joiner = Joiner(aux_table=aux_table, main_key="Countryy", aux_key="country")

    with pytest.raises(
        ValueError,
        match="do not exist in 'X'",
    ):
        false_joiner.fit(main_table)

    false_joiner2 = Joiner(aux_table=aux_table, main_key="Country", aux_key="bad")
    with pytest.raises(
        ValueError,
        match="do not exist in 'aux_table'",
    ):
        false_joiner2.fit(main_table)


@pytest.mark.parametrize("px, assert_frame_equal_", ASSERT_TUPLES)
def test_multiple_keys(px, assert_frame_equal_):
    df = px.DataFrame(
        {"Co": ["France", "Italia", "Deutchland"], "Ca": ["Paris", "Roma", "Berlin"]}
    )
    df2 = px.DataFrame(
        {"CO": ["France", "Italy", "Germany"], "CA": ["Paris", "Rome", "Berlin"]}
    )
    joiner_list = Joiner(
        aux_table=df2,
        aux_key=["CO", "CA"],
        main_key=["Co", "Ca"],
        add_match_info=False,
    )
    result = joiner_list.fit_transform(df)
    try:
        expected = px.concat([df, df2], axis=1)
    except TypeError:
        expected = px.concat([df, df2], how="horizontal")
    assert_frame_equal_(result, expected)

    joiner_list = Joiner(
        aux_table=df2, aux_key="CA", main_key="Ca", add_match_info=False
    )
    result = joiner_list.fit_transform(df)
    assert_frame_equal_(result, expected)


def test_pandas_aux_table_index():
    main_table = pd.DataFrame({"Country": ["France", "Italia", "Spain"]})
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
    assert join["Country_capitals"].tolist() == ["France", "Italy", "Germany"]


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
