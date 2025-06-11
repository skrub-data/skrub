import warnings

import numpy as np
import pytest

from skrub import _dataframe as sbd
from skrub import column_associations


def test_column_associations(df_module):
    x = (np.ones((7, 3)) * np.arange(3)).ravel()
    y = 2 - 3 * x
    z = np.arange(len(x))
    df = df_module.make_dataframe(dict(x=x, y=y, z=z))
    asso = column_associations(df)
    asso = sbd.pandas_convert_dtypes(asso)
    assert sbd.to_list(sbd.col(asso, "left_column_name")) == ["x", "y", "x"]
    assert sbd.to_list(sbd.col(asso, "left_column_idx")) == [0, 1, 0]
    assert sbd.to_list(sbd.col(asso, "right_column_name")) == ["y", "z", "z"]
    assert sbd.to_list(sbd.col(asso, "right_column_idx")) == [1, 2, 2]
    assert sbd.to_list(sbd.col(asso, "cramer_v")) == pytest.approx(
        [1.0, 0.6546536, 0.6546536]
    )
    assert sbd.to_list(sbd.col(asso, "pearson_corr")) == pytest.approx(
        [-1.0, -0.13484, 0.13484]
    )


def test_infinite(df_module):
    # non-regression test for https://github.com/skrub-data/skrub/issues/1133
    # (column associations would raise an exception on low-cardinality float
    # column with infinite values)
    with warnings.catch_warnings():
        # pandas convert_dtypes() emits a spurious warning while trying to decide if
        # floats should be cast to int or not
        # eg `pd.Series([float('inf')]).convert_dtypes()` raises the warning
        warnings.filterwarnings("ignore", message="invalid value encountered in cast")

        column_associations(
            df_module.make_dataframe({"a": [float("inf"), 1.5], "b": [0.0, 1.5]})
        )
