import numpy as np
import pytest

from skrub import _dataframe as sbd
from skrub._drop_if_too_many_nulls import DropIfTooManyNulls


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
            "mixed_null": [None, np.nan, None],
        }
    )


def test_single_column(drop_null_table, df_module):
    # Check that null columns are dropped and non-null columns are kept.
    dn = DropIfTooManyNulls()
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

    # Check that the threshold works
    dn = DropIfTooManyNulls(threshold=0.5)
    assert dn.fit_transform(sbd.col(drop_null_table, "value_nan")) == []
    assert dn.fit_transform(sbd.col(drop_null_table, "value_almost_nan")) == []
    assert dn.fit_transform(sbd.col(drop_null_table, "value_almost_nan")) == []

    df_module.assert_column_equal(
        dn.fit_transform(sbd.col(drop_null_table, "value_mostly_not_nan")),
        df_module.make_column("value_mostly_not_nan", [2.5, 2.5, np.nan]),
    )

    df_module.assert_column_equal(
        dn.fit_transform(sbd.col(drop_null_table, "value_mostly_not_null")),
        df_module.make_column("value_mostly_not_null", ["almost", "almost", None]),
    )

    # Check that setting the threshold to None keeps null columns
    dn = DropIfTooManyNulls(threshold=None)

    df_module.assert_column_equal(
        dn.fit_transform(sbd.col(drop_null_table, "value_nan")),
        df_module.make_column("value_nan", [np.nan, np.nan, np.nan]),
    )

    df_module.assert_column_equal(
        dn.fit_transform(sbd.col(drop_null_table, "value_null")),
        df_module.make_column("value_null", [None, None, None]),
    )

    # Check that setting the threshold to 0 drops columns with at least one
    # null, but keeps columns with no nulls

    dn = DropIfTooManyNulls(threshold=0)

    assert dn.fit_transform(sbd.col(drop_null_table, "value_mostly_not_null")) == []

    df_module.assert_column_equal(
        dn.fit_transform(sbd.col(drop_null_table, "idx")),
        df_module.make_column("idx", [1, 2, 3]),
    )


def test_error_checking(drop_null_table):
    dn = DropIfTooManyNulls(threshold=-1)

    with pytest.raises(ValueError):
        dn.fit_transform(sbd.col(drop_null_table, "value_nan"))
