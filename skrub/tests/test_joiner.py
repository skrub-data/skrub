import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from skrub import Joiner
from skrub._dataframe._polars import POLARS_SETUP
from skrub._dataframe._test_utils import is_module_polars

MODULES = [pd]
ASSERT_TUPLES = [(pd, assert_frame_equal)]

if POLARS_SETUP:
    import polars as pl
    from polars.testing import assert_frame_equal as assert_frame_equal_pl

    MODULES.append(pl)
    ASSERT_TUPLES.append((pl, assert_frame_equal_pl))


@pytest.mark.parametrize("px", MODULES)
def test_joiner(px):
    if is_module_polars(px):
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'")
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
    joiner = Joiner(aux_table=aux_table, main_key="Country", aux_key="country")

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
    if is_module_polars(px):
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'")
    df = px.DataFrame(
        {"Co": ["France", "Italia", "Deutchland"], "Ca": ["Paris", "Roma", "Berlin"]}
    )
    df2 = px.DataFrame(
        {"CO": ["France", "Italy", "Germany"], "CA": ["Paris", "Rome", "Berlin"]}
    )
    joiner_list = Joiner(aux_table=df2, aux_key=["CO", "CA"], main_key=["Co", "Ca"])
    result = joiner_list.fit_transform(df)
    expected = px.DataFrame(px.concat([df, df2], axis=1))
    assert_frame_equal_(result, expected)

    joiner_list = Joiner(aux_table=df2, aux_key="CA", main_key="Ca")
    result = joiner_list.fit_transform(df)
    expected = px.DataFrame(px.concat([df, df2], axis=1))
    assert_frame_equal_(result, expected)
