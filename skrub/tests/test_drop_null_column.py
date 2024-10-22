import numpy as np
import pytest

from skrub import TableVectorizer
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
        }
    )


def test_single_column(drop_null_table, df_module):
    """Check that null columns are dropped and non-null columns are kept."""
    dn = DropNullColumn()
    assert dn.fit_transform(drop_null_table["value_nan"]) == []
    assert dn.fit_transform(drop_null_table["value_null"]) == []

    df_module.assert_column_equal(
        sbd.col(drop_null_table, "idx"), df_module.make_column("idx", [1, 2, 3])
    )

    df_module.assert_column_equal(
        sbd.col(drop_null_table, "value_almost_nan"),
        df_module.make_column("value_almost_nan", [2.5, np.nan, np.nan]),
    )

    df_module.assert_column_equal(
        sbd.col(drop_null_table, "value_almost_null"),
        df_module.make_column("value_almost_null", ["almost", None, None]),
    )


def test_drop_null_column(drop_null_table):
    """Check that all null columns are dropped, and no more."""
    # Don't drop null columns
    tv = TableVectorizer(drop_null_columns=False)
    transformed = tv.fit_transform(drop_null_table)

    assert sbd.shape(transformed) == sbd.shape(drop_null_table)

    # Drop null columns
    tv = TableVectorizer(drop_null_columns=True)
    transformed = tv.fit_transform(drop_null_table)
    assert sbd.shape(transformed) == (sbd.shape(drop_null_table)[0], 3)


def test_is_all_null(drop_null_table):
    """Check that is_all_null is evaluating null counts correctly."""
    # Check that all null columns are marked as "all null"
    assert sbd.is_all_null(drop_null_table["value_nan"])
    assert sbd.is_all_null(drop_null_table["value_null"])

    # Check that the other columns are *not* marked as "all null"
    assert not sbd.is_all_null(drop_null_table["value_almost_null"])
    assert not sbd.is_all_null(drop_null_table["value_almost_nan"])
    assert not sbd.is_all_null(drop_null_table["idx"])
