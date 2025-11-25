"""
Note: most tests in this file use the ``df_module`` fixture, which is defined
in ``skrub.conftest``. See the corresponding docstrings for details.
"""

import inspect
import re
import warnings
from datetime import datetime, timedelta
from functools import partial

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from packaging.version import parse
from pandas.testing import assert_frame_equal as pd_assert_frame_equal

import skrub
from skrub import selectors as s
from skrub._dataframe import _common as ns
from skrub.conftest import skip_polars_installed_without_pyarrow


def test_not_implemented():
    # make codecov happy
    has_default_impl = {
        "is_dataframe",
        "is_column",
        "collect",
        "is_lazyframe",
        "pandas_convert_dtypes",
        "is_column_list",
        "to_column_list",
        "reset_index",
        "copy_index",
        "index",
        "with_columns",
        "select_rows",
    }
    for func_name in sorted(set(ns.__all__) - has_default_impl):
        func = getattr(ns, func_name)
        n_params = len(inspect.signature(func).parameters)
        params = [None] * n_params
        with pytest.raises(TypeError):
            func(*params)
        dop = [skrub.var("a")] * n_params
        with pytest.raises(
            TypeError,
            match=r"Expected a Pandas or Polars .*, but got a skrub DataOp",
        ):
            func(*dop)


#
# Inspecting containers' type and module
# ======================================
#


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


def test_to_list(df_module):
    col = ns.col(df_module.example_dataframe, "str-col")
    if ns.is_pandas(col) and parse(pd.__version__).major >= parse("3.0.0").major:
        # In pandas 3.0, nulls in string dtypes have type np.nan, but nullable dtypes
        # become None
        # To avoid adding even more conditions I'm checking all elements one by one
        #
        to_list = ns.to_list(col)
        assert to_list[0] == "one"
        assert pd.isna(to_list[1])
        assert to_list[2:] == ["three", "four"]
    else:
        assert ns.to_list(col) == ["one", None, "three", "four"]


def test_to_numpy(df_module, example_data_dict):
    array = ns.to_numpy(ns.col(df_module.example_dataframe, "int-col"))
    assert array.dtype == float
    assert_array_equal(array, np.asarray(example_data_dict["int-col"], dtype=float))

    array = ns.to_numpy(ns.col(df_module.example_dataframe, "str-col"))
    assert array.dtype == object
    assert_array_equal(array[2:], np.asarray(example_data_dict["str-col"])[2:])


@skip_polars_installed_without_pyarrow
def test_to_pandas(df_module, pd_module):
    with pytest.raises(TypeError):
        ns.to_pandas(np.arange(3))

    if df_module.name == "pandas":
        assert ns.to_pandas(df_module.example_dataframe) is df_module.example_dataframe
        assert ns.to_pandas(df_module.example_column) is df_module.example_column
        return
    pd_module.assert_frame_equal(
        ns.to_pandas(df_module.example_dataframe).drop(
            ["datetime-col", "date-col"], axis=1
        ),
        pd_module.example_dataframe.drop(["datetime-col", "date-col"], axis=1),
    )
    pd_module.assert_column_equal(
        ns.to_pandas(df_module.example_column),
        pd_module.example_column.astype("float64"),
    )


def test_make_dataframe_like(df_module, example_data_dict):
    df = ns.make_dataframe_like(df_module.empty_dataframe, example_data_dict)
    if df_module.description == "pandas-nullable-dtypes":
        # for pandas, make_dataframe_like will return an old-style / numpy
        # dtypes dataframe
        df = df.convert_dtypes()
    df_module.assert_frame_equal(df, df_module.make_dataframe(example_data_dict))
    assert ns.dataframe_module_name(df) == df_module.name


def test_make_dataframe_like_pandas_index():
    c1 = pd.Series([10, 11], index=[2, 3], name="c1")
    c2 = pd.Series([100, 110], index=[4, 5], name="c2")
    expected = pd.DataFrame({"c1": [10, 11], "c2": [100, 110]})
    df = ns.make_dataframe_like(c1, [c1, c2])
    pd_assert_frame_equal(df, expected)
    df = ns.make_dataframe_like(c1, {"c1": c1, "c2": c2})
    pd_assert_frame_equal(df, expected)


def test_make_column_like(df_module, example_data_dict):
    col = ns.make_column_like(
        df_module.empty_column, example_data_dict["float-col"], "mycol"
    )
    if df_module.description == "pandas-nullable-dtypes":
        # for pandas, make_column_like will return an old-style / numpy dtypes Series
        col = col.convert_dtypes()
    df_module.assert_column_equal(
        col, df_module.make_column(values=example_data_dict["float-col"], name="mycol")
    )
    assert ns.dataframe_module_name(col) == df_module.name

    col = df_module.make_column("old_name", [1, 2, 3])
    expected = df_module.make_column("new_name", [1, 2, 3])
    df_module.assert_column_equal(ns.make_column_like(col, col, "new_name"), expected)


def test_null_value_for(df_module):
    assert ns.null_value_for(df_module.example_dataframe) is None


def test_all_null_like(df_module):
    col = ns.all_null_like(df_module.example_column)
    assert ns.is_column(col)
    assert ns.shape(col) == ns.shape(df_module.example_column)
    expected = df_module.make_column("float-col", [True] * ns.shape(col)[0])
    if df_module.description == "pandas-nullable-dtypes":
        expected = expected.astype("bool")
    df_module.assert_column_equal(ns.is_null(col), expected)

    col = ns.all_null_like(df_module.example_column, length=2)
    assert ns.shape(col)[0] == 2

    # Test can set length greater than column length
    max_len = ns.shape(df_module.example_column)[0]
    col = ns.all_null_like(df_module.example_column, length=max_len + 2)
    assert ns.shape(col)[0] == (max_len + 2)


def test_concat_horizontal(df_module, example_data_dict):
    df1 = df_module.make_dataframe(example_data_dict)
    df2 = ns.set_column_names(df1, list(map("{}1".format, ns.column_names(df1))))
    df = ns.concat(df1, df2, axis=1)
    assert ns.column_names(df) == ns.column_names(df1) + ns.column_names(df2)

    # Test concatenating dataframes with the same column names
    df2 = df1
    df = ns.concat(df1, df2, axis=1)
    assert ns.shape(df) == (4, 16)
    for n in ns.column_names(df)[8:]:
        assert re.match(r".*__skrub_[0-9a-f]+__", n)

    # Test concatenating pandas dataframes with different indexes (of same length)
    if df_module.name == "pandas":
        df1 = df_module.DataFrame(data=[1.0, 2.0], columns=["a"], index=[10, 20])
        df2 = df_module.DataFrame(data=[3.0, 4.0], columns=["b"], index=[1, 2])
        df = ns.concat(df1, df2, axis=1)
        assert ns.shape(df) == (2, 2)
        # Index of the first dataframe is kept
        assert_array_equal(df.index, [10, 20])


def test_concat_vertical(df_module, example_data_dict):
    df1 = df_module.make_dataframe(example_data_dict)
    df2 = ns.set_column_names(df1, list(map("{}1".format, ns.column_names(df1))))
    df = ns.concat(df1, df2, axis=1)
    assert ns.column_names(df) == ns.column_names(df1) + ns.column_names(df2)

    # Test concatenating dataframes with the same column names
    df2 = df_module.make_dataframe(example_data_dict)  # it's a copy with same structure
    df = ns.concat(df1, df2, axis=0)
    assert ns.shape(df) == (8, 8)
    assert ns.column_names(df) == ns.column_names(df1)

    # Test concatenating pandas dataframes with different indexes (of same length)
    if df_module.name == "pandas":
        pd_df1 = pd.DataFrame(data=[1.0, 2.0], columns=["a"], index=[10, 20])
        pd_df2 = pd.DataFrame(
            data=[3.0, 4.0], columns=["a"], index=[1, 2]
        )  # Same columns
        df = ns.concat(pd_df1, pd_df2, axis=0)
        assert ns.shape(df) == (4, 1)

        # Test with overlapping index - should still concatenate
        pd_df3 = pd.DataFrame(data=[5.0, 6.0], columns=["a"], index=[20, 30])
        df = ns.concat(pd_df1, pd_df3, axis=0)
        assert ns.shape(df) == (4, 1)
        if isinstance(df, pd.DataFrame):
            assert_array_equal(
                df.index.to_numpy(),
                np.array(range(len(df))),
            )
        else:
            pass


def test_concat_series(df_module):
    df = df_module.example_dataframe
    col = df_module.example_column

    # Mixing types is not allowed
    msg = r"got dataframes at position \[0\], series at position \[1\]"
    with pytest.raises(TypeError, match=msg):
        ns.concat(df, col)

    msg = r"got dataframes at position \[1\], series at position \[0\]."
    with pytest.raises(TypeError, match=msg):
        ns.concat(col, df)

    msg = (
        r"got dataframes at position \[2\], series at position \[0\], "
        r"types that are neither dataframes nor series at position \[1 3\]"
    )
    with pytest.raises(TypeError, match=msg):
        ns.concat(col, 0, df, 1)

    # Cols only is allowed
    for axis in 0, 1:
        assert (
            ns.shape(ns.concat(col, col, axis=axis))[axis]
            == ns.shape(ns.to_frame(col))[axis] * 2
        )


def test_is_column_list(df_module):
    assert ns.is_column_list([])
    assert ns.is_column_list(())
    assert ns.is_column_list(ns.to_column_list(df_module.example_dataframe))
    assert ns.is_column_list(tuple(ns.to_column_list(df_module.example_dataframe)))
    assert ns.is_column_list([df_module.example_column])

    assert not ns.is_column_list(df_module.example_dataframe)
    assert not ns.is_column_list(df_module.example_column)
    assert not ns.is_column_list([df_module.example_dataframe])
    assert not ns.is_column_list([np.ones(3)])
    assert not ns.is_column_list(np.ones(3))
    assert not ns.is_column_list(np.ones((3, 3)))
    assert not ns.is_column_list(df_module.example_column for i in range(2))
    assert not ns.is_column_list({"col": df_module.example_column})


def test_to_column_list(df_module, example_data_dict):
    cols = ns.to_column_list(df_module.example_dataframe)
    for c, name in zip(cols, example_data_dict.keys()):
        assert ns.name(c) == name
    assert ns.to_column_list(df_module.example_column)[0] is df_module.example_column
    assert ns.to_column_list([df_module.example_column])[0] is df_module.example_column
    assert ns.to_column_list([]) == []
    with pytest.raises(TypeError, match=".*should be a Data.*"):
        ns.to_column_list({"A": [1]})
    with pytest.raises(TypeError, match=".*should be a Data.*"):
        ns.to_column_list(None)


def test_to_column_list_duplicate_columns(pd_module):
    df = pd_module.make_dataframe({"a": [1, 2], "b": [3, 4]})
    df.columns = ["a", "a"]
    col_list = ns.to_column_list(df)
    assert ns.name(col_list[0]) == "a"
    assert ns.to_list(col_list[0]) == [1, 2]
    assert ns.name(col_list[1]) == "a"
    assert ns.to_list(col_list[1]) == [3, 4]


def test_collect(df_module):
    assert ns.collect(df_module.example_dataframe) is df_module.example_dataframe
    if df_module.name == "polars":
        df_module.assert_frame_equal(
            ns.collect(df_module.example_dataframe.lazy()), df_module.example_dataframe
        )


def test_col(df_module):
    assert ns.to_list(ns.col(df_module.example_dataframe, "float-col"))[0] == 4.5


def test_col_by_idx(df_module):
    assert ns.name(ns.col_by_idx(df_module.example_dataframe, 2)) == "float-col"


def test_col_by_idx_duplicate_columns(pd_module):
    df = pd_module.make_dataframe({"a": [1, 2], "b": [3, 4]})
    df.columns = ["a", "a"]
    assert ns.to_list(ns.col_by_idx(df, 0)) == [1, 2]


#
# Querying, modifying metadata and shape
# ===============================
#


def test_shape(df_module):
    assert ns.shape(df_module.example_dataframe) == (4, 8)
    assert ns.shape(df_module.empty_dataframe) == (0, 0)
    assert ns.shape(df_module.example_column) == (4,)
    assert ns.shape(df_module.empty_column) == (0,)


def test_to_frame(df_module):
    col = df_module.example_column
    assert ns.is_dataframe(ns.to_frame(col))


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


def test_reset_index(df_module):
    df, col = df_module.example_dataframe, df_module.example_column
    if df_module.name != "pandas":
        assert ns.reset_index(df) is df
        assert ns.reset_index(col) is col
        return
    idx = [-10 - i for i in range(len(col))]

    df1 = df.set_axis(idx, axis="index")
    assert df1.index.tolist() == idx
    assert ns.reset_index(df1).index.tolist() == list(range(df1.shape[0]))
    assert df1.index.tolist() == idx

    col1 = col.set_axis(idx)
    assert col1.index.tolist() == idx
    assert ns.reset_index(col1).index.tolist() == list(range(len(col1)))
    assert col1.index.tolist() == idx


@pytest.mark.parametrize("source_is_df", [False, True])
@pytest.mark.parametrize("target_is_df", [False, True])
def test_copy_index(source_is_df, target_is_df):
    source = pd.Series(list("abc"), index=[10, 20, 30], name="s")
    if source_is_df:
        source = pd.DataFrame({"s": source}, index=source.index)
    target = pd.Series(list("def"), index=[100, 200, 300], name="t")
    if target_is_df:
        target = pd.DataFrame({"t": target}, index=target.index)
    out = ns.copy_index(source, target)
    assert (out.index == source.index).all()
    assert target.index.tolist() == [100, 200, 300]
    assert source.index.tolist() == [10, 20, 30]


def test_copy_index_list(pd_module):
    a = pd_module.example_column
    b = []
    assert ns.copy_index(a, b) is b
    assert ns.copy_index(b, a) is a


def test_copy_index_polars(pd_module, pl_module):
    a = pd_module.example_column
    b = pl_module.example_column
    assert ns.copy_index(a, b) is b
    assert ns.copy_index(b, a) is a


def test_index(df_module):
    df, col = df_module.example_dataframe, df_module.example_column
    if df_module.name == "pandas":
        assert ns.index(df) is df.index
        assert ns.index(col) is col.index
    else:
        assert ns.index(df) is None
        assert ns.index(col) is None


#
# Inspecting dtypes and casting
# =============================
#


def test_dtype(df_module):
    df = df_module.example_dataframe
    assert ns.dtype(ns.col(df, "float-col")) == df_module.dtypes["float64"]
    assert ns.dtype(ns.col(df, "int-not-null-col")) == df_module.dtypes["int64"]


def test_dtypes(df_module):
    df = df_module.example_dataframe
    assert ns.dtypes(s.select(df, ["int-not-null-col", "float-col"])) == [
        df_module.dtypes["int64"],
        df_module.dtypes["float64"],
    ]


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


def test_is_integer(df_module):
    df = df_module.example_dataframe
    assert ns.is_integer(ns.col(df, "int-not-null-col"))
    if df_module.description != "pandas-numpy-dtypes":
        assert ns.is_integer(ns.col(df, "int-col"))
    for col in [
        "float-col",
        "str-col",
        "datetime-col",
        "date-col",
        "bool-col",
        "bool-not-null-col",
    ]:
        assert not ns.is_integer(ns.col(df, col))


def test_is_float(df_module):
    df = df_module.example_dataframe
    assert ns.is_float(ns.col(df, "float-col"))
    for col in [
        "int-not-null-col",
        "str-col",
        "datetime-col",
        "date-col",
        "bool-col",
        "bool-not-null-col",
    ]:
        assert not ns.is_float(ns.col(df, col))


def test_is_string(df_module):
    df = df_module.example_dataframe
    df = ns.pandas_convert_dtypes(df)
    assert ns.is_string(ns.col(df, "str-col"))
    for col in ["int-col", "float-col", "datetime-col", "date-col", "bool-col"]:
        assert not ns.is_string(ns.col(df, col))


def test_sentinel_is_string_pandas_3(df_module):
    if df_module.name != "pandas":
        return
    pd_version = parse(pd.__version__)
    if pd_version.major < parse("3.0.0").major:
        return
    if not pd_version.is_prerelease:
        pytest.fail("This test should fail when pandas 3.x is released.")


def test_to_string(df_module):
    s = ns.to_string(df_module.make_column("_", list(range(5))))
    assert ns.is_string(s)


def test_to_string_polars_object(pl_module):
    s = pl_module.make_column("", [object()])
    ns.to_string(s)


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
        assert ns.is_any_date(ns.col(df, date_col))
    for col in ["str-col", "int-col", "float-col", "bool-col"]:
        assert not ns.is_any_date(ns.col(df, col))


def test_to_datetime(df_module):
    s = df_module.make_column("", ["01/02/2020", "02/01/2021", "bad"])
    with pytest.raises(ValueError):
        ns.to_datetime(s, "%m/%d/%Y", True)
    assert ns.to_list(ns.to_datetime(s, "%m/%d/%Y", False)) == ns.to_list(
        df_module.make_column("", [datetime(2020, 1, 2), datetime(2021, 2, 1), None])
    )
    assert ns.to_list(ns.to_datetime(s, "%d/%m/%Y", False)) == ns.to_list(
        df_module.make_column("", [datetime(2020, 2, 1), datetime(2021, 1, 2), None])
    )
    dt_col = ns.col(df_module.example_dataframe, "datetime-col")
    assert ns.to_datetime(dt_col, None) is dt_col
    s = df_module.make_column("", ["2020-01-01 04:00:00+02:00"])
    dt = ns.to_datetime(s, "%Y-%m-%d %H:%M:%S%z")
    assert str(dt[0]) == "2020-01-01 02:00:00+00:00"


def test_is_duration(df_module):
    df = df_module.make_dataframe(
        {"a": [timedelta(days=1)], "b": [datetime(2020, 3, 4)]}
    )
    assert ns.is_duration(ns.col(df, "a"))
    assert not ns.is_duration(ns.col(df, "b"))


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
    if df_module.description == "pandas-numpy-dtypes":
        import pandas as pd

        assert s.dtype == pd.CategoricalDtype(list("ab"))
        assert list(s.cat.categories) == list("ab")

    if df_module.description == "pandas-nullable-dtypes":
        import pandas as pd

        assert s.dtype == pd.CategoricalDtype(pd.Series(list("ab")).astype("string"))
        assert list(s.cat.categories) == list("ab")


def test_is_all_null(df_module):
    """Check that is_all_null is evaluating null counts correctly."""

    # Check that all null columns are marked as "all null"
    assert ns.is_all_null(df_module.make_column("all_null", [None, None, None]))
    assert ns.is_all_null(df_module.make_column("all_nan", [np.nan, np.nan, np.nan]))
    assert ns.is_all_null(
        df_module.make_column("all_nan_or_null", [np.nan, np.nan, None])
    )

    # Check that the other columns are *not* marked as "all null"
    assert not ns.is_all_null(
        df_module.make_column("almost_all_null", ["almost", None, None])
    )
    assert not ns.is_all_null(
        df_module.make_column("almost_all_nan", [2.5, None, None])
    )


def test_is_all_null_polars(pl_module):
    """Special case for polars: column is full of nulls, but doesn't have dtype Null"""
    col = pl_module.make_column("col", [1, None, None])
    col = col[1:]

    assert ns.is_all_null(col)


# Inspecting, selecting and modifying values
# ==========================================
#


@pytest.mark.parametrize(
    "values, expected",
    [
        ([False, True, None], False),
        ([False, True, True], False),
        ([True, True, None], True),
    ],
)
def test_all(values, expected, df_module):
    s = df_module.make_column("", values)
    assert ns.all(s) == expected


@pytest.mark.parametrize(
    "values, expected",
    [
        ([False, True, None], True),
        ([False, False, None], False),
        ([False, False, False], False),
    ],
)
def test_any(values, expected, df_module):
    s = df_module.make_column("", values)
    assert ns.any(s) == expected


def test_sum(df_module):
    assert ns.sum(df_module.example_column) == np.nansum(
        ns.to_numpy(df_module.example_column)
    )


def test_min(df_module):
    assert ns.min(df_module.example_column) == np.nanmin(
        ns.to_numpy(df_module.example_column)
    )


def test_max(df_module):
    assert ns.max(df_module.example_column) == np.nanmax(
        ns.to_numpy(df_module.example_column)
    )


def test_std(df_module):
    assert float(ns.std(df_module.example_column)) == pytest.approx(
        float(np.nanstd(ns.to_numpy(df_module.example_column), ddof=1))
    )


def test_mean(df_module):
    assert ns.mean(df_module.example_column) == np.nanmean(
        ns.to_numpy(df_module.example_column)
    )


@skip_polars_installed_without_pyarrow
def test_corr(df_module):
    df = df_module.example_dataframe

    # Make sure we use Pandas to compute Pearson's correlation.
    expected_corr = ns.to_pandas(df).corr(numeric_only=True)
    corr = ns.copy_index(expected_corr, ns.to_pandas(ns.pearson_corr(df)))

    pd.testing.assert_frame_equal(corr, expected_corr)


@pytest.mark.parametrize(
    "descending, expected_vals",
    [
        (False, [3, 1, 4, 2]),
        (True, [4, 1, 3, 2]),
    ],
)
def test_sort(df_module, descending, expected_vals):
    df = df_module.make_dataframe({"a": [2.0, None, 1.0, 3.0], "b": [1, 2, 3, 4]})
    sorted_b = ns.col(ns.sort(df, by="a", descending=descending), "b")
    expected_b = df_module.make_column("b", expected_vals)
    df_module.assert_column_equal(sorted_b, expected_b)


def test_value_counts(df_module):
    col = df_module.make_column("x", ["a", "b", "a", None, None, "a"])
    counts = ns.value_counts(col)
    counts = ns.sort(counts, by="value")
    expected = df_module.make_dataframe({"value": ["a", "b"], "count": [3, 1]})
    expected = ns.sort(expected, by="value")
    if ns.is_pandas(col) and parse(pd.__version__).major >= parse("3.0.0").major:
        # Added to avoid a failing type check since we don't care about dtype here
        assert (ns.to_numpy(expected) == ns.to_numpy(counts)).all()
    else:
        df_module.assert_frame_equal(counts, expected)


@pytest.mark.parametrize("q", [0.0, 0.3, 1.0])
@pytest.mark.parametrize(
    "interpolation", ["nearest", "higher", "lower", "midpoint", "linear"]
)
def test_quantile(df_module, q, interpolation):
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    x[::10] = np.nan
    col = df_module.make_column("x", x)
    assert ns.quantile(col, q, interpolation=interpolation) == np.nanquantile(
        x, q, method=interpolation
    )


@pytest.mark.parametrize("obj", ["column", "dataframe"])
@pytest.mark.parametrize("s", [(1, 1), (0, 3), (-1, None), (None, 3), (None, None)])
def test_slice(df_module, obj, s):
    out = ns.slice(getattr(df_module, f"example_{obj}"), *s)
    if obj == "dataframe":
        out = ns.col(out, "float-col")
    out = ns.to_numpy(out)
    expected = ns.to_numpy(df_module.example_column)[slice(*s)]
    assert_array_equal(out, expected)


@pytest.mark.parametrize("obj", ["column", "dataframe"])
@pytest.mark.parametrize("idx", [[], [1], [2, 1]])
def test_select_rows(df_module, obj, idx):
    out = ns.select_rows(getattr(df_module, f"example_{obj}"), idx)
    if obj == "dataframe":
        out = ns.col(out, "float-col")
    out = ns.to_numpy(out)
    expected = ns.to_numpy(df_module.example_column)[idx]
    assert_array_equal(out, expected)


def test_select_rows_array():
    a = np.arange(6).reshape((3, 2))
    assert_array_equal(ns.select_rows(a, (2, 1)), a[[2, 1], :])
    assert_array_equal(ns.select_rows(a[0], (1, 0)), a[0, [1, 0]])


def test_is_null(df_module):
    s = ns.pandas_convert_dtypes(df_module.make_column("", [0, None, 2, None, 4]))
    expected = df_module.make_column("", [False, True, False, True, False])
    if df_module.description == "pandas-nullable-dtypes":
        expected = expected.astype("bool")
    df_module.assert_column_equal(ns.is_null(s), expected)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0, 1, None], True),
        ([10, 10, None], True),
        ([0, 0, 0], False),
    ],
)
def test_has_nulls(values, expected, df_module):
    s = df_module.make_column("", values)
    assert ns.has_nulls(s) == expected


def test_drop_nulls(df_module):
    s = ns.pandas_convert_dtypes(df_module.make_column("", [0, None, 2, None, 4]))
    df_module.assert_column_equal(
        ns.drop_nulls(s),
        ns.pandas_convert_dtypes(df_module.make_column("", [0, 2, 4])),
    )


def test_fill_nulls(df_module):
    # Test on dataframe
    df = df_module.make_dataframe({"col_1": [0, np.nan, 2], "col_2": [0, None, 2.0]})
    df_module.assert_frame_equal(
        ns.fill_nulls(df, -1),
        df_module.make_dataframe({"col_1": [0.0, -1, 2.0], "col_2": [0.0, -1.0, 2.0]}),
    )

    # Test on series
    s = ns.pandas_convert_dtypes(
        df_module.make_column("", [0.0, np.nan, 2.0, None, 4.0])
    )
    df_module.assert_column_equal(
        ns.fill_nulls(s, -1),
        ns.pandas_convert_dtypes(
            df_module.make_column("", [0.0, -1.0, 2.0, -1.0, 4.0])
        ),
    )


def test_unique(df_module):
    s = ns.pandas_convert_dtypes(df_module.make_column("", [0, None, 2, None, 4]))
    assert ns.n_unique(s) == 3
    df_module.assert_column_equal(
        ns.unique(s), ns.pandas_convert_dtypes(df_module.make_column("", [0, 2, 4]))
    )


def test_filter(df_module):
    df = df_module.example_dataframe
    pred = ns.col(df, "int-not-null-col") > 1
    filtered_df = ns.filter(df, pred)
    assert ns.shape(filtered_df) == (2, 8)
    assert ns.to_list(ns.col(filtered_df, "int-not-null-col")) == [4, 10]
    filtered_col = ns.filter(df_module.example_column, pred)
    assert ns.to_list(filtered_col) == [4.5, -1.5]


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


def test_where_row(df_module):
    df = df_module.make_dataframe({"col1": [1, 2, 3], "col2": [1000, 2000, 3000]})
    out = ns.where_row(
        df,
        df_module.make_column("", [False, True, False]),  # mask
        df_module.make_column(
            "", [None, None, None]
        ),  # values to put in on the entire row
    )
    right = df_module.make_dataframe(
        {"col1": [None, 2, None], "col2": [None, 2000, None]}
    )
    df_module.assert_frame_equal(
        ns.pandas_convert_dtypes(out),
        ns.pandas_convert_dtypes(right),
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


def test_nans_treated_as_nulls(df_module):
    "Non-regression test for https://github.com/skrub-data/skrub/issues/916"
    col = partial(df_module.make_column, "")

    def same(c1, c2):
        return df_module.assert_column_equal(
            ns.pandas_convert_dtypes(c1), ns.pandas_convert_dtypes(c2)
        )

    with warnings.catch_warnings():
        # pandas warning when it checks if a column that contains inf could be
        # cast to int
        warnings.simplefilter("ignore")
        s = col([1.1, None, 2.2, float("nan"), float("inf")])
        same(ns.is_null(s), col([False, True, False, True, False]))

        same(ns.drop_nulls(s), col([1.1, 2.2, float("inf")]))
        same(ns.fill_nulls(s, -1.0), col([1.1, -1.0, 2.2, -1.0, float("inf")]))


def test_with_columns(df_module):
    df = df_module.make_dataframe({"a": [1, 2], "b": [3, 4]})

    # Add one new col
    out = ns.with_columns(df, **{"c": [5, 6]})
    if df_module.description == "pandas-nullable-dtypes":
        # for pandas, make_column_like will return an old-style / numpy dtypes Series
        out = ns.pandas_convert_dtypes(out)
    expected = df_module.make_dataframe({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    df_module.assert_frame_equal(out, expected)

    # Add multiple new cols
    out = ns.with_columns(df, **{"c": [5, 6], "d": [7, 8]})
    if df_module.description == "pandas-nullable-dtypes":
        out = ns.pandas_convert_dtypes(out)
    expected = df_module.make_dataframe(
        {"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]}
    )
    df_module.assert_frame_equal(out, expected)

    # Pass a col instead of an array
    out = ns.with_columns(df, **{"c": df_module.make_column("c", [5, 6])})
    if df_module.description == "pandas-nullable-dtypes":
        out = ns.pandas_convert_dtypes(out)
    expected = df_module.make_dataframe({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    df_module.assert_frame_equal(out, expected)

    # Replace col
    out = ns.with_columns(df, **{"a": [5, 6]})
    if df_module.description == "pandas-nullable-dtypes":
        out = ns.pandas_convert_dtypes(out)
    expected = df_module.make_dataframe({"a": [5, 6], "b": [3, 4]})
    df_module.assert_frame_equal(out, expected)


def test_abs(df_module):
    s = df_module.make_column("", [-1.0, 2.0, None])
    df_module.assert_column_equal(
        ns.abs(s), df_module.make_column("", [1.0, 2.0, None])
    )


def test_total_seconds(df_module):
    s = df_module.make_column("", [timedelta(seconds=20), timedelta(hours=1)])
    assert ns.to_list(ns.total_seconds(s)) == [20, 3600]


@pytest.mark.parametrize(
    "col, expected",
    [
        ([1, 2, 3], True),
        (["a", "b", "c"], True),
        ([1, 3, 2], False),
        (["a", "c", "b"], False),
        ([1, None, 3], True),
        ([inspect, re, ns.is_sorted], False),  # weird object dtype
    ],
)
def test_is_sorted(col, expected, df_module):
    col = df_module.make_column("", col)
    assert ns.is_sorted(col) == expected
    if expected:
        assert not ns.is_sorted(col[::-1])
        assert ns.is_sorted(col[::-1], descending=True)


@pytest.mark.parametrize(
    "col", [[[1, 2], [3, 4]], [{"a": 1, "b": 2}, {"a": 1, "b": 3}]]
)
def test_is_sorted_object_dtypes(col, df_module):
    # For those more complex dtypes where the result is more ambiguous pandas &
    # polars or even different versions of the same package can disagree on
    # whether they are sorted. For the time being we don't have a strong reason
    # to add the code / computation time to handle those discrepancies.
    # However, is_sorted should not crash and return a Boolean in all cases.
    assert isinstance(ns.is_sorted(df_module.make_column("", col)), bool)
