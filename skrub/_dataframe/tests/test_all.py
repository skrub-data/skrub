import pytest

from skrub._dataframe import Selector, skrubns


@pytest.fixture
def df(px):
    return px.DataFrame(
        {"ID": [2, 3, 7], "name": ["ab", "cd", "01"], "temp": [20.3, 40.9, 11.5]}
    )


def test_select(df):
    ns = skrubns(df)
    assert list(ns.select(df, []).columns) == []
    assert list(ns.select(df, ["name"]).columns) == ["name"]
    assert list(ns.select(df, Selector.ALL).columns) == list(df.columns)
    assert list(ns.select(df, Selector.NONE).columns) == []
    assert list(ns.select(df, Selector.NUMERIC).columns) == ["ID", "temp"]
    assert list(ns.select(df, Selector.CATEGORICAL).columns) == ["name"]


def test_drop(df):
    ns = skrubns(df)
    assert list(ns.drop(df, []).columns) == list(df.columns)
    assert list(ns.drop(df, ["name"]).columns) == ["ID", "temp"]
    assert list(ns.drop(df, Selector.ALL).columns) == []
    assert list(ns.drop(df, Selector.NONE).columns) == list(df.columns)
    assert list(ns.drop(df, Selector.NUMERIC).columns) == ["name"]
    assert list(ns.drop(df, Selector.CATEGORICAL).columns) == ["ID", "temp"]


def test_concat_horizontal(df):
    ns = skrubns(df)
    df1 = (
        df.__dataframe_consortium_standard__()
        .rename_columns({c: f"{c}_1" for c in df.columns})
        .dataframe
    )
    out = ns.concat_horizontal(df)
    assert list(out.columns) == list(df.columns)
    out = ns.concat_horizontal(df, df1)
    assert list(out.columns) == list(df.columns) + list(df1.columns)
