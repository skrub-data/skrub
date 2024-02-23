"""
Note: most tests in this file use the ``df_module`` fixture, which is defined
in ``skrub.conftest``. See the corresponding docstrings for details.
"""

from datetime import datetime

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


def test_dtype(df_module):
    df = ns.pandas_convert_dtypes(df_module.example_dataframe)
    assert ns.dtype(ns.col(df, "float-col")) == df_module.dtypes["float64"]
    assert ns.dtype(ns.col(df, "int-col")) == df_module.dtypes["int64"]


def test_cast(df_module):
    col = ns.col(df_module.example_dataframe, "int-col")
    out = ns.cast(col, df_module.dtypes["float64"])
    assert ns.dtype(out) == df_module.dtypes["float64"]


def test_pandas_convert_dtypes(df_module):
    if df_module.name == "pandas":
        df_module.assert_frame_equal(
            ns.pandas_convert_dtypes(df_module.example_dataframe),
            df_module.example_dataframe.convert_dtypes(),
        )
        df_module.assert_column_equal(
            ns.pandas_convert_dtypes(df_module.example_column),
            df_module.example_column.convert_dtypes(),
        )
    else:
        assert (
            ns.pandas_convert_dtypes(df_module.example_dataframe)
            is df_module.example_dataframe
        )
        assert (
            ns.pandas_convert_dtypes(df_module.example_column)
            is df_module.example_column
        )


def test_is_bool(df_module):
    df = df_module.example_dataframe
    df = ns.pandas_convert_dtypes(df)
    assert ns.is_bool(ns.col(df, "bool-col"))
    assert not ns.is_bool(ns.col(df, "int-col"))


def test_is_numeric(df_module):
    df = df_module.example_dataframe
    df = ns.pandas_convert_dtypes(df)
    for num_col in ["int-col", "float-col"]:
        assert ns.is_numeric(ns.col(df, num_col))
    for col in ["str-col", "datetime-col", "date-col", "bool-col"]:
        assert not ns.is_numeric(ns.col(df, col))


def test_to_numeric(df_module):
    s = ns.to_string(df_module.make_column("_", list(range(5))))
    assert ns.is_string(s)
    as_num = ns.to_numeric(s)
    assert ns.is_numeric(as_num)
    assert ns.dtype(as_num) == df_module.dtypes["int64"]
    df_module.assert_column_equal(
        as_num, ns.pandas_convert_dtypes(df_module.make_column("_", list(range(5))))
    )


def test_is_string(df_module):
    df = df_module.example_dataframe
    df = ns.pandas_convert_dtypes(df)
    assert ns.is_string(ns.col(df, "str-col"))
    for col in ["int-col", "float-col", "datetime-col", "date-col", "bool-col"]:
        assert not ns.is_string(ns.col(df, col))


def test_to_string(df_module):
    s = ns.to_string(df_module.make_column("_", list(range(5))))
    assert ns.is_string(s)


def test_is_object(df_module):
    if df_module.name == "polars":
        import polars as pl

        s = pl.Series("", [1, "abc"], dtype=pl.Object)
    else:
        s = df_module.make_column("", [1, "abc"])
    assert ns.is_object(ns.pandas_convert_dtypes(s))

    s = df_module.make_column("", ["1", "abc"])
    assert not ns.is_object(ns.pandas_convert_dtypes(s))


def test_is_anydate(df_module):
    df = df_module.example_dataframe
    df = ns.pandas_convert_dtypes(df)
    date_cols = ["datetime-col"]
    if df_module.name != "pandas":
        # pandas does not have a Date type
        date_cols.append("date-col")
    for date_col in date_cols:
        assert ns.is_anydate(ns.col(df, date_col))
    for col in ["str-col", "int-col", "float-col", "bool-col"]:
        assert not ns.is_anydate(ns.col(df, col))


def test_to_datetime(df_module):
    s = df_module.make_column("", ["01/02/2020", "02/01/2021", "bad"])
    with pytest.raises(ValueError):
        ns.to_datetime(s, "%m/%d/%Y", True)
    df_module.assert_column_equal(
        ns.to_datetime(s, "%m/%d/%Y", False),
        df_module.make_column("", [datetime(2020, 1, 2), datetime(2021, 2, 1), None]),
    )
    df_module.assert_column_equal(
        ns.to_datetime(s, "%d/%m/%Y", False),
        df_module.make_column("", [datetime(2020, 2, 1), datetime(2021, 1, 2), None]),
    )
    s = df_module.make_column("", ["2020-01-02", "2021-04-05"])
    df_module.assert_column_equal(
        ns.to_datetime(s, None, True),
        df_module.make_column("", [datetime(2020, 1, 2), datetime(2021, 4, 5)]),
    )


def test_is_categorical(df_module):
    if df_module.name == "pandas":
        import pandas as pd

        s = pd.Series(list("aab"))
        assert not ns.is_categorical(s)
        s = pd.Series(list("aab"), dtype="category")
        assert ns.is_categorical(s)
    elif df_module.name == "polars":
        import polars as pl

        s = pl.Series(list("aab"))
        assert not ns.is_categorical(s)
        s = pl.Series(list("aab"), dtype=pl.Categorical)
        assert ns.is_categorical(s)
        s = pl.Series(list("aab"), dtype=pl.Enum("ab"))
        assert ns.is_categorical(s)


def test_to_categorical(df_module):
    s = df_module.make_column("", list("aab"))
    assert not ns.is_categorical(s)
    s = ns.to_categorical(s)
    assert ns.is_categorical(s)
    if df_module.name == "polars":
        import polars as pl

        assert s.dtype == pl.Categorical
        assert list(s.cat.get_categories()) == list("ab")
    if df_module.name == "pandas":
        import pandas as pd

        assert s.dtype == pd.CategoricalDtype(list("ab"))


#
# Inspecting, selecting and modifying values
# ==========================================
#


def test_is_in(df_module):
    s = df_module.make_column("", list("aabc") + ["", None])
    s = ns.pandas_convert_dtypes(s)
    df_module.assert_column_equal(
        ns.is_in(s, list("ac")),
        ns.pandas_convert_dtypes(
            df_module.make_column("", [True, True, False, True, False, None])
        ),
    )


def test_is_null(df_module):
    s = ns.pandas_convert_dtypes(df_module.make_column("", [0, None, 2, None, 4]))
    df_module.assert_column_equal(
        ns.is_null(s), df_module.make_column("", [False, True, False, True, False])
    )


def test_drop_nulls(df_module):
    s = ns.pandas_convert_dtypes(df_module.make_column("", [0, None, 2, None, 4]))
    df_module.assert_column_equal(
        ns.drop_nulls(s),
        ns.pandas_convert_dtypes(df_module.make_column("", [0, 2, 4])),
    )


def test_unique(df_module):
    s = ns.pandas_convert_dtypes(df_module.make_column("", [0, None, 2, None, 4]))
    assert ns.n_unique(s) == 3
    df_module.assert_column_equal(
        ns.unique(s), ns.pandas_convert_dtypes(df_module.make_column("", [0, 2, 4]))
    )


def test_where(df_module):
    s = ns.pandas_convert_dtypes(df_module.make_column("", [0, 1, 2]))
    out = ns.where(
        s,
        df_module.make_column("", [True, False, True]),
        df_module.make_column("", [10, 11, 12]),
    )
    df_module.assert_column_equal(
        out, ns.pandas_convert_dtypes(df_module.make_column("", [0, 11, 2]))
    )


def test_sample(df_module):
    s = ns.pandas_convert_dtypes(df_module.make_column("", [0, 1, 2]))
    sample = ns.sample(s, 2)
    assert ns.shape(sample)[0] == 2
    vals = set(ns.to_numpy(sample))
    assert len(vals) == 2
    assert vals.issubset([0, 1, 2])


def test_replace(df_module):
    s = ns.pandas_convert_dtypes(
        df_module.make_column("", "aa ab ac ba bb bc".split() + [None])
    )
    out = ns.replace(s, "ac", "AC")
    expected = ns.pandas_convert_dtypes(
        df_module.make_column("", "aa ab AC ba bb bc".split() + [None])
    )
    df_module.assert_column_equal(out, expected)

    out = ns.replace_regex(s, "^a", r"A_")
    expected = ns.pandas_convert_dtypes(
        df_module.make_column("", "A_a A_b A_c ba bb bc".split() + [None])
    )
    df_module.assert_column_equal(out, expected)
