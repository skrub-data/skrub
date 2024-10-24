import numpy as np
import pytest

from skrub import _dataframe as sbd
from skrub._drop_null_column import DropNullColumn


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


def test_single_column(drop_null_table, df_module):
    """Check that null columns are dropped and non-null columns are kept."""
    dn = DropNullColumn()
    assert dn.fit_transform(drop_null_table["value_nan"]) == []
    assert dn.fit_transform(drop_null_table["value_null"]) == []
    assert dn.fit_transform(drop_null_table["mixed_null"]) == []

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
