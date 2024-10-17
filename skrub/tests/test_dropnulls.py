import numpy as np
import pandas as pd
import polars as pl
import pytest

from numpy.testing import assert_array_equal

from skrub import TableVectorizer
from skrub._dataframe import _common as ns
from skrub._dropnull import DropNullColumn


@pytest.fixture
def main_table(df_module):
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


def test_drop_nullcolumn(main_table):

    dn = DropNullColumn()
    assert dn.fit_transform(main_table["value_nan"]) == []
    assert dn.fit_transform(main_table["value_null"]) == []

    assert_array_equal(ns.to_numpy(ns.col(main_table, "idx")), np.array([1, 2, 3]))

def test_transform_nullcolumn(main_table):
    main_table_dropped = ns.drop(main_table, "value_null")
    main_table_dropped = ns.drop(main_table_dropped, "value_nan")
    
    # Don't drop null columns
    tv = TableVectorizer(drop_null_columns=False)
    transformed = tv.fit_transform(main_table)

    assert ns.shape(transformed) == ns.shape(main_table)
    
    # Drop null columns
    tv = TableVectorizer(drop_null_columns=True)
    transformed = tv.fit_transform(main_table)
    assert ns.shape(transformed) == (ns.shape(main_table)[0], 1)

