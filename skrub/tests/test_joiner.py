import pandas as pd
import pytest
from sklearn.utils._testing import assert_array_equal

from skrub import Joiner
from skrub.dataframe.tests.test_polars import (
    POLARS_MISSING_MSG,
    POLARS_SETUP,
    XFAIL_POLARS,
)


@pytest.mark.skipif(not POLARS_SETUP, reason=POLARS_MISSING_MSG)
@pytest.mark.xfail(
    "joiner" in XFAIL_POLARS, reason="Polars not supported for joiner yet."
)
def test_polars_input() -> None:
    import polars as pl

    main_data = {
        "Country": [
            "France",
            "Germany",
            "Italy",
        ]
    }

    aux_1_data = {
        "Country": ["Germany", "French Republic", "Italia"],
        "Population": [84_000_000, 68_000_000, 59_000_000],
    }

    aux_2_data = {
        "Country name": ["France", "Italy", "Germany", "UK"],
        "GDP (billion):": [2937, 2099, 4223, 3186],
    }

    aux_3_data = {
        "Countries": ["La France", "Italy", "Germany"],
        "Capital": ["Paris", "Rome", "Berlin"],
    }

    main_table_pd = pd.DataFrame(main_data)
    aux_table_1_pd = pd.DataFrame(aux_1_data)
    aux_table_2_pd = pd.DataFrame(aux_2_data)
    aux_table_3_pd = pd.DataFrame(aux_3_data)

    aux_tables_pd = [
        (aux_table_1_pd, "Country"),
        (aux_table_2_pd, "Country name"),
        (aux_table_3_pd, "Countries"),
    ]

    main_table_pl = pl.DataFrame(main_data)
    aux_table_1_pl = pl.DataFrame(aux_1_data)
    aux_table_2_pl = pl.DataFrame(aux_2_data)
    aux_table_3_pl = pl.DataFrame(aux_3_data)

    aux_tables_pl = [
        (aux_table_1_pl, "Country"),
        (aux_table_2_pl, "Country name"),
        (aux_table_3_pl, "Countries"),
    ]

    joiner = Joiner(tables=aux_tables_pd, main_key="Country")
    joiner.fit(main_table_pd)
    big_table_pd = joiner.transform(main_table_pd)

    joiner = Joiner(tables=aux_tables_pl, main_key="Country")
    joiner.fit(main_table_pl)
    big_table_pl = joiner.transform(main_table_pl)

    assert_array_equal(big_table_pd, big_table_pl)


def test_joiner() -> None:
    main_table = pd.DataFrame(
        [
            "France",
            "Germany",
            "Italy",
        ],
        columns=["Country"],
    )

    aux_table_1 = pd.DataFrame(
        [
            ["Germany", 84_000_000],
            ["French Republic", 68_000_000],
            ["Italia", 59_000_000],
        ],
        columns=["Country", "Population"],
    )

    aux_table_2 = pd.DataFrame(
        [
            ["France", 2937],
            ["Italy", 2099],
            ["Germany", 4223],
            ["UK", 3186],
        ],
        columns=["Country name", "GDP (billion)"],
    )

    aux_table_3 = pd.DataFrame(
        [
            ["La France", "Paris"],
            ["Italy", "Rome"],
            ["Germany", "Berlin"],
        ],
        columns=["Countries", "Capital"],
    )

    aux_tables = [
        (aux_table_1, "Country"),
        (aux_table_2, "Country name"),
        (aux_table_3, "Countries"),
    ]

    joiner = Joiner(tables=aux_tables, main_key="Country")

    joiner.fit(main_table)

    number_of_cols = tuple(
        map(
            sum,
            zip(
                main_table.shape,
                aux_table_1.shape,
                aux_table_2.shape,
                aux_table_3.shape,
            ),
        )
    )[1]

    big_table = joiner.transform(main_table)
    assert big_table.shape == (main_table.shape[0], number_of_cols)

    big_table = joiner.fit_transform(main_table)
    assert big_table.shape == (main_table.shape[0], number_of_cols)

    false_joiner = Joiner(tables=aux_tables, main_key="Countryy")

    with pytest.raises(
        ValueError,
        match=r"Main key",
    ):
        false_joiner.fit(main_table)

    false_aux_tables = [
        (aux_table_1, ["Countrys"]),
        (aux_table_2, "Country name"),
        (aux_table_3, "Countries"),
    ]

    false_joiner2 = Joiner(tables=false_aux_tables, main_key="Country")
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
    joiner_list = Joiner(tables=[(df2, ["CO", "CA"])], main_key=["Co", "Ca"])
    result = joiner_list.fit_transform(df)
    expected = pd.DataFrame(pd.concat([df, df2], axis=1))
    pd.testing.assert_frame_equal(result, expected)

    # Equivalent signature with tuples:
    joiner_tuple = Joiner(tables=(df2, ["CO", "CA"]), main_key=["Co", "Ca"])
    result = joiner_tuple.fit_transform(df)
    expected = pd.DataFrame(pd.concat([df, df2], axis=1))
    pd.testing.assert_frame_equal(result, expected)

    joiner_list = Joiner(tables=(df2, "CA"), main_key="Ca")
    result = joiner_list.fit_transform(df)
    expected = pd.DataFrame(pd.concat([df, df2], axis=1))
    pd.testing.assert_frame_equal(result, expected)
