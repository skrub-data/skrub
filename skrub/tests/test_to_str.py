import numpy as np
import pandas as pd
import pytest

from skrub import _dataframe as sbd
from skrub._on_each_column import RejectColumn
from skrub._to_categorical import ToCategorical
from skrub._to_datetime import ToDatetime
from skrub._to_str import ToStr


def test_to_str(df_module):
    values = [object(), True, None, "one"]
    out = ToStr().fit_transform(df_module.make_column("", values))
    assert sbd.is_string(out)
    if df_module.name == "pandas":
        assert out.dtype == "O"
        assert pd.NA not in out
    expected = df_module.make_column(
        "", [None if v is None else str(v) for v in values]
    )
    if df_module.description == "pandas-nullable-dtypes":
        null = expected.isna()
        expected = expected.astype("str")
        expected[null] = None
    df_module.assert_column_equal(sbd.drop_nulls(out), sbd.drop_nulls(expected))
    df_module.assert_column_equal(sbd.is_null(out), sbd.is_null(expected))


def test_rejected_columns(df_module):
    columns = [
        ToDatetime().fit_transform(df_module.make_column("", ["2020-02-02"])),
        ToCategorical().fit_transform(df_module.make_column("", ["a", "b"])),
    ]
    for col in columns:
        with pytest.raises(RejectColumn):
            ToStr().fit_transform(col)
        to_str = ToStr().fit(df_module.make_column("", [""]))
        assert sbd.is_string(to_str.transform(col))


def test_pandas_string():
    s = pd.Series(["a", "b"], dtype="string")
    assert ToStr().fit_transform(s).dtype == "O"


def test_pandas_na():
    s = pd.Series(["a", pd.NA], dtype="str")
    assert s[1] is pd.NA
    out = ToStr().fit_transform(s)
    assert out[1] is not pd.NA
    assert np.isnan(out[1])
