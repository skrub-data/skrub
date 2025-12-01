import numpy as np
import pytest

from skrub import _dataframe as sbd
from skrub._single_column_transformer import RejectColumn
from skrub._to_categorical import ToCategorical
from skrub._to_datetime import ToDatetime
from skrub._to_float import ToFloat, _str_replace
from skrub.conftest import skip_polars_installed_without_pyarrow


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
def test_to_float(values, df_module):
    s = df_module.make_column("c", values)
    out = ToFloat().fit_transform(s)
    assert is_float32(df_module, out)


@skip_polars_installed_without_pyarrow
def test_rejected_columns(df_module):
    columns = [
        df_module.make_column("c", ["1", "2", "hello"]),
        ToDatetime().fit_transform(df_module.make_column("c", ["2020-02-02"])),
        ToCategorical().fit_transform(df_module.make_column("c", ["1", "2"])),
    ]
    for col in columns:
        with pytest.raises(RejectColumn):
            ToFloat().fit_transform(col)
        to_float = ToFloat().fit(df_module.make_column("c", [1.1]))
        assert is_float32(df_module, to_float.transform(col))


@pytest.mark.parametrize(
    "input_str, expected_float, decimal",
    [
        ("1,234.56", 1234.56, "."),
        ("1.234,56", 1234.56, ","),
        ("1 234,56", 1234.56, ","),
        ("1234.56", 1234.56, "."),
        ("1234,56", 1234.56, ","),
        ("1,234,567.89", 1234567.89, "."),
        ("1.234.567,89", 1234567.89, ","),
        ("1 234 567,89", 1234567.89, ","),
        ("1'234'567.89", 1234567.89, "."),
        ("1.23e+4", 12300.0, "."),
        ("1.23E+4", 12300.0, "."),
        ("1,23e+4", 12300.0, ","),
        ("1,23E+4", 12300.0, ","),
        ("-1,234.56", -1234.56, "."),
        ("-1.234,56", -1234.56, ","),
        ("(1,234.56)", -1234.56, "."),
        ("(1.234,56)", -1234.56, ","),
        ("1,23,456.78", 123456.78, "."),
        ("12,3456.78", 123456.78, "."),
        (".56", 0.56, "."),
        (",56", 0.56, ","),
    ],
)
def test_number_parsing(input_str, expected_float, decimal, df_module):
    column = df_module.make_column("col", [input_str])
    result = ToFloat(decimal=decimal).fit_transform(column)

    np.allclose(result[0], expected_float)


def test_str_replace(df_module):
    s = df_module.make_column("x", ["1,234.56", "7.890,12", "3 456,78", "9'012.34"])
    result_dot = _str_replace(s, decimal=".")

    expected_dot = df_module.make_column(
        "x", ["1234.56", "7890.12", "3456.78", "9012.34"]
    )
    np.all(sbd.to_list(result_dot) == sbd.to_list(expected_dot))

    with pytest.raises(TypeError):
        _str_replace([1, 2, 3], decimal=".")
