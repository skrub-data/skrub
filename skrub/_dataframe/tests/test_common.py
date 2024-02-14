import numpy as np
import pytest
from numpy.testing import assert_array_equal

from skrub._dataframe import _common as ns

#
# Inspecting containers' type and module
# ======================================
#


def test_skrub_namespace(df_module):
    skrub_ns = ns.skrub_namespace(df_module.empty_dataframe)
    assert skrub_ns.DATAFRAME_MODULE_NAME == df_module.name


def test_dataframe_module_name(df_module):
    assert ns.dataframe_module_name(df_module.empty_dataframe) == df_module.name
    assert getattr(ns, f"is_{df_module.name}")(df_module.empty_dataframe)
    assert ns.dataframe_module_name(df_module.empty_column) == df_module.name
    assert getattr(ns, f"is_{df_module.name}")(df_module.empty_column)


def test_is_dataframe(df_module):
    assert ns.is_dataframe(df_module.empty_dataframe)
    assert not ns.is_dataframe(df_module.empty_column)
    assert not ns.is_dataframe(np.eye(3))
    assert not ns.is_dataframe({"a": [1, 2]})


def test_is_lazyframe(df_module):
    assert not ns.is_lazyframe(df_module.empty_dataframe)
    if hasattr(df_module, "empty_lazyframe"):
        assert ns.is_lazyframe(df_module.empty_lazyframe)


def test_is_column(df_module):
    assert ns.is_column(df_module.empty_column)
    assert not ns.is_column(df_module.empty_dataframe)
    assert not ns.is_column(np.eye(3))
    assert not ns.is_column({"a": [1, 2]})


#
# Conversions to and from other container types
# =============================================
#


def test_to_numpy(df_module, example_data_dict):
    with pytest.raises(NotImplementedError):
        ns.to_numpy(df_module.example_dataframe)
    array = ns.to_numpy(ns.col(df_module.example_dataframe, "int-col"))
    assert array.dtype == float
    assert_array_equal(array, np.asarray(example_data_dict["int-col"], dtype=float))

    array = ns.to_numpy(ns.col(df_module.example_dataframe, "str-col"))
    assert array.dtype == object
    assert_array_equal(array, np.asarray(example_data_dict["str-col"]))


def test_to_pandas(df_module, all_dataframe_modules):
    pd_module = all_dataframe_modules["pandas"]
    if df_module.name == "pandas":
        assert ns.to_pandas(df_module.example_dataframe) is df_module.example_dataframe
        assert ns.to_pandas(df_module.example_column) is df_module.example_column
    pd_module.assert_frame_equal(
        ns.to_pandas(df_module.example_dataframe).drop(
            ["datetime-col", "date-col"], axis=1
        ),
        pd_module.example_dataframe.drop(["datetime-col", "date-col"], axis=1),
    )
    pd_module.assert_column_equal(
        ns.to_pandas(df_module.example_column), pd_module.example_column
    )

    with pytest.raises(NotImplementedError):
        ns.to_pandas(np.arange(3))


def test_make_dataframe_like(df_module, example_data_dict):
    df = ns.make_dataframe_like(df_module.empty_dataframe, example_data_dict)
    df_module.assert_frame_equal(df, df_module.make_dataframe(example_data_dict))
    assert ns.dataframe_module_name(df) == df_module.name


def test_make_column_like(df_module, example_data_dict):
    col = ns.make_column_like(
        df_module.empty_column, example_data_dict["float-col"], "mycol"
    )
    df_module.assert_column_equal(
        col, df_module.make_column(values=example_data_dict["float-col"], name="mycol")
    )
    assert ns.dataframe_module_name(col) == df_module.name


#
# Querying and modifying metadata
# ===============================
#


def test_shape(df_module):
    assert ns.shape(df_module.example_dataframe) == (4, 6)
    assert ns.shape(df_module.empty_dataframe) == (0, 0)
    assert ns.shape(df_module.example_column) == (4,)
    assert ns.shape(df_module.empty_column) == (0,)


@pytest.mark.parametrize("name", ["", "a\nname"])
def test_name(df_module, name):
    assert ns.name(df_module.make_column(name=name, values=[0])) == name


def test_column_names(df_module, example_data_dict):
    col_names = ns.column_names(df_module.example_dataframe)
    assert isinstance(col_names, list)
    assert col_names == list(example_data_dict.keys())
    assert ns.column_names(df_module.empty_dataframe) == []


def test_rename(df_module):
    col = df_module.make_column(name="name", values=[0])
    col1 = ns.rename(col, "name 1")
    assert ns.name(col) == "name"
    assert ns.name(col1) == "name 1"


def test_set_column_names(df_module, example_data_dict):
    df = df_module.make_dataframe(example_data_dict)
    old_names = ns.column_names(df)
    new_names = list(map("col_{}".format, range(ns.shape(df)[1])))
    new_df = ns.set_column_names(df, new_names)
    assert ns.column_names(df) == old_names
    assert ns.column_names(new_df) == new_names


#
# Inspecting dtypes and casting
# =============================
#

#
# Inspecting, selecting and modifying values
# ==========================================
#
