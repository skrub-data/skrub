import numpy as np
import pytest

from skrub import _dataframe as sbd
from skrub._drop_column_if_null import DropColumnIfNull
from skrub._on_each_column import RejectColumn


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
            "mixed_null": [None, np.nan, None],
        }
    )


def test_single_column_drop(drop_null_table, df_module):
    """Check that null columns are dropped and non-null columns are kept."""
    dn = DropColumnIfNull(null_column_strategy="drop")
    assert dn.fit_transform(sbd.col(drop_null_table, "value_nan")) == []
    assert dn.fit_transform(sbd.col(drop_null_table, "value_null")) == []
    assert dn.fit_transform(sbd.col(drop_null_table, "mixed_null")) == []

    df_module.assert_column_equal(
        dn.fit_transform(sbd.col(drop_null_table, "idx")),
        df_module.make_column("idx", [1, 2, 3]),
    )

    df_module.assert_column_equal(
        dn.fit_transform(sbd.col(drop_null_table, "value_almost_nan")),
        df_module.make_column("value_almost_nan", [2.5, np.nan, np.nan]),
    )

    df_module.assert_column_equal(
        dn.fit_transform(sbd.col(drop_null_table, "value_almost_null")),
        df_module.make_column("value_almost_null", ["almost", None, None]),
    )


def test_single_column_keep(drop_null_table, df_module):
    """Check that all columns are kept."""
    dn = DropColumnIfNull(null_column_strategy="keep")

    df_module.assert_column_equal(
        dn.fit_transform(sbd.col(drop_null_table, "idx")),
        df_module.make_column("idx", [1, 2, 3]),
    )

    df_module.assert_column_equal(
        dn.fit_transform(sbd.col(drop_null_table, "value_null")),
        df_module.make_column(
            "value_null",
            [
                None,
                None,
                None,
            ],
        ),
    )

    df_module.assert_column_equal(
        dn.fit_transform(sbd.col(drop_null_table, "value_nan")),
        df_module.make_column(
            "value_nan",
            [
                np.nan,
                np.nan,
                np.nan,
            ],
        ),
    )

    df_module.assert_column_equal(
        dn.fit_transform(sbd.col(drop_null_table, "mixed_null")),
        df_module.make_column("mixed_null", [None, np.nan, None]),
    )

    df_module.assert_column_equal(
        dn.fit_transform(sbd.col(drop_null_table, "value_almost_nan")),
        df_module.make_column("value_almost_nan", [2.5, np.nan, np.nan]),
    )

    df_module.assert_column_equal(
        dn.fit_transform(sbd.col(drop_null_table, "value_almost_null")),
        df_module.make_column("value_almost_null", ["almost", None, None]),
    )


def test_single_column_raise(drop_null_table, df_module):
    """Check that an exception is raised if a null column is detected."""
    dn = DropColumnIfNull(null_column_strategy="raise")
    with pytest.raises(RejectColumn):
        dn.fit_transform(sbd.col(drop_null_table, "value_nan"))
    with pytest.raises(RejectColumn):
        dn.fit_transform(sbd.col(drop_null_table, "value_null"))
    with pytest.raises(RejectColumn):
        dn.fit_transform(sbd.col(drop_null_table, "mixed_null"))


def test_incorrect_argument():
    with pytest.raises(ValueError):
        DropColumnIfNull(null_column_strategy="wrong value")
