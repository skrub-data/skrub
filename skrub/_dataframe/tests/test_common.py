import numpy as np
import pytest
from numpy.testing import assert_array_equal

from skrub._dataframe import _common as ns

#
# Inspecting containers' type and module
# ======================================
#


def test_skrub_namespace(empty_df):
    skrub_ns = ns.skrub_namespace(empty_df)
    assert skrub_ns.DATAFRAME_MODULE_NAME == ns.dataframe_module_name(empty_df)


def test_dataframe_module_name_pandas(empty_pandas_dataframe, empty_pandas_series):
    assert ns.dataframe_module_name(empty_pandas_dataframe) == "pandas"
    assert ns.is_pandas(empty_pandas_dataframe)
    assert ns.dataframe_module_name(empty_pandas_series) == "pandas"
    assert ns.is_pandas(empty_pandas_series)


def test_dataframe_module_name_polars(empty_polars_dataframe, empty_polars_series):
    assert ns.dataframe_module_name(empty_polars_dataframe) == "polars"
    assert ns.is_polars(empty_polars_dataframe)
    assert ns.dataframe_module_name(empty_polars_series) == "polars"
    assert ns.is_polars(empty_polars_series)


def test_is_dataframe_df(empty_df):
    assert ns.is_dataframe(empty_df)


def test_is_dataframe_col(empty_column):
    assert not ns.is_dataframe(empty_column)


@pytest.mark.parametrize("obj", ["a", [1, 2], np.arange(5), np.ones(3)])
def test_is_dataframe_other(obj):
    assert not ns.is_dataframe(obj)


def test_is_lazyframe(empty_polars_dataframe):
    assert not ns.is_lazyframe(empty_polars_dataframe)
    assert ns.is_lazyframe(empty_polars_dataframe.lazy())


def test_is_column_col(empty_column):
    assert ns.is_column(empty_column)


def test_is_column_df(empty_df):
    assert not ns.is_column(empty_df)


@pytest.mark.parametrize("obj", ["a", [1, 2], np.arange(5), np.ones(3)])
def test_is_column_other(obj):
    assert not ns.is_column(obj)


#
# Conversions to and from other container types
# =============================================
#


def test_to_numpy(example_df):
    with pytest.raises(NotImplementedError):
        ns.to_numpy(example_df)
    array = ns.to_numpy(ns.col(example_df, "int-col"))
    assert array.dtype == float
    assert_array_equal(array, [4.0, 0.0, -1.0, np.nan])

    array = ns.to_numpy(ns.col(example_df, "str-col"))
    assert array.dtype == object
    assert_array_equal(array, ["one", None, "three", "four"])


#
# Querying and modifying metadata
# ===============================
#


def test_column_names(empty_df):
    data = {"a": [0], "b c": [0]}
    df = ns.make_dataframe_like(empty_df, data)
    assert ns.column_names(df) == ["a", "b c"]


#
# Inspecting dtypes and casting
# =============================
#

#
# Inspecting, selecting and modifying values
# ==========================================
#
