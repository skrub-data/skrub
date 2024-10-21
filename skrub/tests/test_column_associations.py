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
