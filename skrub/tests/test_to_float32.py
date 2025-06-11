import numpy as np
import pytest

from skrub import _dataframe as sbd
from skrub._on_each_column import RejectColumn
from skrub._to_categorical import ToCategorical
from skrub._to_datetime import ToDatetime
from skrub._to_float32 import ToFloat32


def is_float32(df_module, column):
    if df_module.name == "pandas":
        return sbd.dtype(column) == np.float32
    return sbd.dtype(column) == df_module.dtypes["float32"]


@pytest.mark.parametrize(
    "values",
    [
        ["1.1", "2.2", None],
        [1.1, 2.2, None],
        [1, 2, 3],
        [True, False, True],
        [True, False, None],
    ],
)
def test_to_float_32(values, df_module):
    s = df_module.make_column("c", values)
    out = ToFloat32().fit_transform(s)
    assert is_float32(df_module, out)


def test_rejected_columns(df_module):
    columns = [
        df_module.make_column("c", ["1", "2", "hello"]),
        ToDatetime().fit_transform(df_module.make_column("c", ["2020-02-02"])),
        ToCategorical().fit_transform(df_module.make_column("c", ["1", "2"])),
    ]
    for col in columns:
        with pytest.raises(RejectColumn):
            ToFloat32().fit_transform(col)
        to_float = ToFloat32().fit(df_module.make_column("c", [1.1]))
        assert is_float32(df_module, to_float.transform(col))
