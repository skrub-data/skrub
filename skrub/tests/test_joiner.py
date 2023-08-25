import pandas as pd
import pytest

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
