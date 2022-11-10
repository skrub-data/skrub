import pandas as pd
import pytest

from dirty_cat import FeatureAugmenter


def test_feature_augmenter():
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
            ["France", 68_000_000],
            ["Italy", 59_000_000],
        ],
        columns=["Country", "Population"],
    )

    aux_table_2 = pd.DataFrame(
        [
            ["France", 2937],
            ["Italy", 2099],
            ["Germany", 4223],
        ],
        columns=["Country name", "GDP (billion)"],
    )

    aux_table_3 = pd.DataFrame(
        [
            ["France", "Paris"],
            ["Italy", "Rome"],
            ["Germany", "Berlin"],
        ],
        columns=["Countries", "Capital"],
    )

    aux_tables = {
        "Country": aux_table_1,
        "Country name": aux_table_2,
        "Countries": aux_table_3,
    }

    fa = FeatureAugmenter(tables=aux_tables, main_key="Country")

    fa.fit(main_table)

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

    big_table = fa.transform(main_table)
    assert big_table.shape == (main_table.shape[0], number_of_cols)

    big_table = fa.fit_transform(main_table)
    assert big_table.shape == (main_table.shape[0], number_of_cols)

    false_fa = FeatureAugmenter(tables=aux_tables, main_key="Countryy")

    with pytest.raises(
        ValueError,
        match=r"column missing in the main",
    ):
        false_fa.fit(main_table)

    false_aux_tables = {
        "Countrys": aux_table_1,
        "Country name": aux_table_2,
        "Countries": aux_table_3,
    }

    false_fa2 = FeatureAugmenter(tables=false_aux_tables, main_key="Country")
    with pytest.raises(
        ValueError,
        match=r"column missing in the auxilliary",
    ):
        false_fa2.fit(main_table)
