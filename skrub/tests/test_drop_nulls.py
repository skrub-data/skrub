import numpy as np
import pytest
from numpy.testing import assert_array_equal

from skrub import TableVectorizer
from skrub._drop_null import DropNullColumn
from skrub import _dataframe as sbd


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
        }
    )


def test_single_column(drop_null_table):
    dn = DropNullColumn()
    assert dn.fit_transform(drop_null_table["value_nan"]) == []
    assert dn.fit_transform(drop_null_table["value_null"]) == []

    assert_array_equal(sbd.to_numpy(sbd.col(drop_null_table, "idx")), np.array([1, 2, 3]))


def test_drop_null_column(drop_null_table):
    main_table_dropped = sbd.drop(drop_null_table, "value_null")
    main_table_dropped = sbd.drop(main_table_dropped, "value_nan")

    # Don't drop null columns
    tv = TableVectorizer(drop_null_columns=False)
    transformed = tv.fit_transform(drop_null_table)

    assert sbd.shape(transformed) == sbd.shape(drop_null_table)

    # Drop null columns
    tv = TableVectorizer(drop_null_columns=True)
    transformed = tv.fit_transform(drop_null_table)
    assert sbd.shape(transformed) == (sbd.shape(drop_null_table)[0], 1)
