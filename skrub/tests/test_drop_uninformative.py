import numpy as np
import pytest

from skrub import _dataframe as sbd
from skrub._drop_uninformative import DropUninformative


@pytest.fixture
def drop_null_table(df_module):
    return df_module.make_dataframe(
        {
            "idx": [
                1,
                2,
                3,
            ],
            "value_nan": [
                np.nan,
                np.nan,
                np.nan,
            ],
            "value_null": [
                None,
                None,
                None,
            ],
            "value_almost_nan": [
                2.5,
                np.nan,
                np.nan,
            ],
            "value_almost_null": [
                "almost",
                None,
                None,
            ],
            "value_mostly_not_nan": [
                2.5,
                2.5,
                np.nan,
            ],
            "value_mostly_not_null": [
                "almost",
                "almost",
                None,
            ],
        }
    )


@pytest.mark.parametrize(
    "params, column, result",
    [
        (dict(), "idx", [1, 2, 3]),
        (dict(), "value_nan", []),
        (dict(), "value_null", []),
        (dict(), "value_almost_nan", [2.5, np.nan, np.nan]),
        (dict(), "value_almost_null", ["almost", None, None]),
        (dict(), "value_mostly_not_nan", [2.5, 2.5, np.nan]),
        (dict(), "value_mostly_not_null", ["almost", "almost", None]),
        (dict(null_fraction_threshold=0.5), "idx", [1, 2, 3]),
        (dict(null_fraction_threshold=0.5), "value_nan", []),
        (dict(null_fraction_threshold=0.5), "value_null", []),
        (dict(null_fraction_threshold=0.5), "value_almost_nan", []),
        (dict(null_fraction_threshold=0.5), "value_almost_null", []),
        (dict(null_fraction_threshold=0.5), "value_mostly_not_nan", [2.5, 2.5, np.nan]),
        (
            dict(null_fraction_threshold=0.5),
            "value_mostly_not_null",
            ["almost", "almost", None],
        ),
    ],
)
def test_drop_nulls(df_module, drop_null_table, params, column, result):
    enc = DropUninformative(**params)
    res = enc.fit_transform(drop_null_table[column])
    if result == []:
        assert res == result
    else:
        df_module.assert_column_equal(res, df_module.make_column(column, result))


def test_do_not_drop_nulls(df_module, drop_null_table):
    enc = DropUninformative(null_fraction_threshold=None)
    for col in drop_null_table.columns:
        res = enc.fit_transform(drop_null_table[col])
        df_module.assert_column_equal(res, drop_null_table[col])


def test_error_checking(drop_null_table):
    dn = DropUninformative(null_fraction_threshold=-1)
    with pytest.raises(ValueError):
        dn.fit_transform(sbd.col(drop_null_table, "value_nan"))

    dn = DropUninformative(drop_if_id="wrong")
    with pytest.raises(ValueError, match="constant_column"):
        dn.fit_transform(sbd.col(drop_null_table, "value_nan"))
    dn = DropUninformative(drop_if_id="wrong")
    with pytest.raises(ValueError, match="column_is_id"):
        dn.fit_transform(sbd.col(drop_null_table, "value_nan"))
    dn = DropUninformative(null_fraction_threshold="wrong")
    with pytest.raises(ValueError, match="Threshold"):
        dn.fit_transform(sbd.col(drop_null_table, "value_nan"))


@pytest.fixture
def drop_constant_table(df_module):
    return df_module.make_dataframe(
        {
            "idx": [
                1,
                2,
                3,
            ],
            "constant_float": [2.5, 2.5, 2.5],
            "constant_float_with_nulls": [
                2.5,
                2.5,
                np.nan,
            ],
            "constant_str": [
                "const",
                "const",
                "const",
            ],
            "constant_str_with_nulls": [
                "const",
                "const",
                None,
            ],
        }
    )


@pytest.mark.parametrize(
    "params, column, result",
    [
        (dict(), "idx", [1, 2, 3]),
        (dict(), "constant_float", []),
        (dict(), "constant_float_with_nulls", [2.5, 2.5, np.nan]),
        (dict(), "constant_str", []),
        (dict(), "constant_str_with_nulls", ["const", "const", None]),
        (dict(constant_column=False), "idx", [1, 2, 3]),
        (dict(constant_column=False), "constant_float", [2.5] * 3),
        (dict(constant_column=False), "constant_float_with_nulls", [2.5, 2.5, np.nan]),
        (dict(constant_column=False), "constant_str", ["const"] * 3),
        (
            dict(constant_column=False),
            "constant_str_with_nulls",
            ["const", "const", None],
        ),
    ],
)
def test_drop_constants(df_module, drop_constant_table, params, column, result):
    enc = DropUninformative(**params)
    res = enc.fit_transform(drop_constant_table[column])
    if result == []:
        assert res == result
    else:
        df_module.assert_column_equal(res, df_module.make_column(column, result))


@pytest.fixture
def drop_id_column(df_module):
    return df_module.make_dataframe(
        {
            "idx": [
                1,
                2,
                3,
            ],
            "idx_with_nulls": [
                1,
                2,
                np.nan,
            ],
            "str": [
                "i1",
                "i2",
                "i3",
            ],
            "str_with_nulls": [
                "i1",
                "i2",
                None,
            ],
            "variable": ["A", "B", "B"],
        }
    )


@pytest.mark.parametrize("drop_ids", [True, False])
def test_drop_id(df_module, drop_id_column, drop_ids):
    enc = DropUninformative(drop_if_id=drop_ids)
    for column in drop_id_column.columns:
        res = enc.fit_transform(drop_id_column[column])
        # Check that "str" is the only column that gets dropped
        if column == "str" and drop_ids:
            assert res == []
        else:
            df_module.assert_column_equal(res, df_module.make_column(column, res))
