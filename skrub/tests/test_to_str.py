import numpy as np
import pandas as pd
import pytest
from packaging.version import parse

from skrub import _dataframe as sbd
from skrub._apply_to_cols import RejectColumn
from skrub._to_datetime import ToDatetime
from skrub._to_str import ToStr
from skrub.conftest import skip_polars_installed_without_pyarrow


def test_to_str(df_module):
    values = [object(), 17, None, "one"]
    out = ToStr().fit_transform(df_module.make_column("", values))
    assert sbd.is_string(out)
    if df_module.name == "pandas":
        if sbd.is_pandas(out) and parse(pd.__version__).major >= parse("3.0.0").major:
            assert out.dtype == pd.StringDtype(na_value=np.nan)
        else:
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


@skip_polars_installed_without_pyarrow
def test_rejected_columns(df_module):
    columns = [
        ToDatetime().fit_transform(df_module.make_column("", ["2020-02-02"])),
        sbd.to_categorical(df_module.make_column("", ["a", "b"])),
    ]
    for col in columns:
        with pytest.raises(RejectColumn):
            ToStr().fit_transform(col)
        to_str = ToStr().fit(df_module.make_column("", [""]))
        assert sbd.is_string(to_str.transform(col))


def test_pandas_string():
    s = pd.Series(["a", "b"], dtype="string")

    if sbd.is_pandas(s) and parse(pd.__version__).major >= parse("3.0.0").major:
        assert ToStr().fit_transform(s).dtype == pd.StringDtype(na_value=np.nan)
    else:
        assert ToStr().fit_transform(s).dtype == "O"


def test_pandas_na():
    s = pd.Series(["a", pd.NA], dtype="str")
    if sbd.is_pandas(s) and parse(pd.__version__).major >= parse("3.0.0").major:
        assert s[1] is np.nan
    else:
        assert s[1] is pd.NA
    out = ToStr().fit_transform(s)
    # This check is not needed with pandas 3.0
    assert out[1] is not pd.NA
    assert np.isnan(out[1])


def test_convert_category(df_module):
    col = sbd.to_categorical(df_module.make_column("", ["a", "b"]))

    with pytest.raises(RejectColumn):
        ToStr().fit_transform(col)

    # force conversion
    transformed = ToStr(convert_category=True).fit_transform(col)
    assert sbd.is_string(transformed)
