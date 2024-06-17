import pandas
import pandas.testing
import pytest

from skrub import DropCols, SelectCols
from skrub._dataframe import _common as ns
from skrub._select_cols import Drop


@pytest.fixture
def df(df_module):
    return df_module.make_dataframe({"A": [1, 2], "B": [10, 20], "C": ["x", "y"]})


def test_select_cols(df):
    selector = SelectCols(["C", "A"])
    out = selector.fit_transform(df)
    assert list(ns.column_names(df)) == ["A", "B", "C"]
    assert list(ns.column_names(out)) == ["C", "A"]
    assert list(ns.col(out, "A")) == [1, 2]
    assert list(ns.col(out, "C")) == ["x", "y"]


def test_select_single_col(df):
    out_1 = SelectCols("A").fit_transform(df)
    out_2 = SelectCols(["A"]).fit_transform(df)
    pandas.testing.assert_frame_equal(pandas.DataFrame(out_1), pandas.DataFrame(out_2))


def test_select_missing_cols(df):
    selector = SelectCols(["X", "A"])
    with pytest.raises(ValueError, match=r".*missing from dataframe: \['X'\]"):
        selector.fit(df)
    df_subset = SelectCols(["C", "A"]).fit_transform(df)
    selector = SelectCols(["A", "B"]).fit(df)
    with pytest.raises(ValueError, match=r".*missing from dataframe: \['B'\]"):
        selector.transform(df_subset)


def test_drop_cols(df):
    selector = DropCols(["C", "A"])
    out = selector.fit_transform(df)
    assert list(ns.column_names(df)) == ["A", "B", "C"]
    assert list(ns.column_names(out)) == ["B"]
    assert list(ns.col(out, "B")) == [10, 20]


def test_drop_single_col(df):
    out_1 = DropCols("A").fit_transform(df)
    out_2 = DropCols(["A"]).fit_transform(df)
    pandas.testing.assert_frame_equal(pandas.DataFrame(out_1), pandas.DataFrame(out_2))


def test_drop_missing_cols(df):
    selector = DropCols(["X", "A"])
    with pytest.raises(ValueError, match=r".*missing from dataframe: \['X'\]"):
        selector.fit(df)


def test_drop(df_module):
    col = df_module.example_column
    assert Drop().fit_transform(col) == []
    assert Drop().fit(col).transform(col) == []
