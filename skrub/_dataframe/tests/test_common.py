"""
Note: most tests in this file use the ``df_module`` fixture, which is defined
in ``skrub.conftest``. See the corresponding docstrings for details.
"""

import inspect
import warnings
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal as pd_assert_frame_equal

from skrub import _selectors as s
from skrub._dataframe import _common as ns


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
    }
    for func_name in sorted(set(ns.__all__) - has_default_impl):
        func = getattr(ns, func_name)
        n_params = len(inspect.signature(func).parameters)
        params = [None] * n_params
        with pytest.raises(NotImplementedError):
            func(*params)


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
    assert ns.to_list(col) == ["one", None, "three", "four"]


def test_to_numpy(df_module, example_data_dict):
    array = ns.to_numpy(ns.col(df_module.example_dataframe, "int-col"))
    assert array.dtype == float
    assert_array_equal(array, np.asarray(example_data_dict["int-col"], dtype=float))

    array = ns.to_numpy(ns.col(df_module.example_dataframe, "str-col"))
    assert array.dtype == object
    assert_array_equal(array[2:], np.asarray(example_data_dict["str-col"])[2:])


def test_to_pandas(df_module, all_dataframe_modules):
    with pytest.raises(NotImplementedError):
        ns.to_pandas(np.arange(3))

    pd_module = all_dataframe_modules["pandas-numpy-dtypes"]
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


def test_concat_horizontal(df_module, example_data_dict):
    df1 = df_module.make_dataframe(example_data_dict)
    df2 = ns.set_column_names(df1, list(map("{}1".format, ns.column_names(df1))))
    df = ns.concat_horizontal(df1, df2)
    assert ns.column_names(df) == ns.column_names(df1) + ns.column_names(df2)


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
    assert not ns.is_column_list((df_module.example_column for i in range(2)))
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


def test_collect(df_module):
    assert ns.collect(df_module.example_dataframe) is df_module.example_dataframe
    if df_module.name == "polars":
        df_module.assert_frame_equal(
            ns.collect(df_module.example_dataframe.lazy()), df_module.example_dataframe
        )


#
# Querying and modifying metadata
# ===============================
#


def test_shape(df_module):
    assert ns.shape(df_module.example_dataframe) == (4, 8)
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


def test_copy_index_non_pandas(all_dataframe_modules):
    a = all_dataframe_modules["pandas-numpy-dtypes"].example_column
    b = []
    assert ns.copy_index(a, b) is b
    assert ns.copy_index(b, a) is a
    if "polars" not in all_dataframe_modules:
        pytest.skip(reason="polars not installed")
    b = all_dataframe_modules["polars"].example_column
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
        assert ns.is_any_date(ns.col(df, date_col))
    for col in ["str-col", "int-col", "float-col", "bool-col"]:
        assert not ns.is_any_date(ns.col(df, col))


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
    dt_col = ns.col(df_module.example_dataframe, "datetime-col")
    assert ns.to_datetime(dt_col, None) is dt_col
    s = df_module.make_column("", ["2020-01-01 04:00:00+02:00"])
    dt = ns.to_datetime(s, "%Y-%m-%d %H:%M:%S%z")
    assert str(dt[0]) == "2020-01-01 02:00:00+00:00"


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
