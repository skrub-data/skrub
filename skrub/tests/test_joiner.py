import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from skrub import Joiner


def test_joiner() -> None:
    main_table = pd.DataFrame(
        [
            "France",
            "Germany",
            "Italy",
        ],
        columns=["Country"],
    )

    aux_table = pd.DataFrame(
        [
            ["Germany", 84_000_000],
            ["French Republic", 68_000_000],
            ["Italia", 59_000_000],
        ],
        columns=["country", "Population"],
    )
    joiner = Joiner(aux_table=aux_table, main_key="Country", aux_key="country")

    joiner.fit(main_table)
    big_table = joiner.transform(main_table)
    assert big_table.shape == (main_table.shape[0], 3)
    assert_array_equal(
        big_table["Population"].values, aux_table["Population"].values[[1, 0, 2]]
    )

    false_joiner = Joiner(aux_table=aux_table, main_key="Countryy", aux_key="country")

    with pytest.raises(
        ValueError,
        match=r"Main key",
    ):
        false_joiner.fit(main_table)

    false_joiner2 = Joiner(aux_table=aux_table, main_key="Country", aux_key="bad")
    with pytest.raises(
        ValueError,
        match=r"Column key",
    ):
        false_joiner2.fit(main_table)


def test_multiple_keys():
    df = pd.DataFrame(
        [["France", "Paris"], ["Italia", "Roma"], ["Deutchland", "Berlin"]],
        columns=["Co", "Ca"],
    )
    df2 = pd.DataFrame(
        [["France", "Paris"], ["Italy", "Rome"], ["Germany", "Berlin"]],
        columns=["CO", "CA"],
    )
    joiner_list = Joiner(aux_table=df2, aux_key=["CO", "CA"], main_key=["Co", "Ca"])
    result = joiner_list.fit_transform(df)
    expected = pd.DataFrame(pd.concat([df, df2], axis=1))
    pd.testing.assert_frame_equal(result, expected)

    joiner_list = Joiner(aux_table=df2, aux_key="CA", main_key="Ca")
    result = joiner_list.fit_transform(df)
    expected = pd.DataFrame(pd.concat([df, df2], axis=1))
    pd.testing.assert_frame_equal(result, expected)
