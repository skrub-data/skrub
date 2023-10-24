import pandas
import pytest

from skrub import DropCols, SelectCols
from skrub.dataframe import POLARS_SETUP

DATAFRAME_MODULES = [pandas]
if POLARS_SETUP:
    import polars

    DATAFRAME_MODULES.append(polars)


@pytest.fixture(params=DATAFRAME_MODULES)
def df(request):
    return request.param.DataFrame({"A": [1, 2], "B": [10, 20], "C": ["x", "y"]})


def test_select_cols(df):
    selector = SelectCols(["C", "A"])
    out = selector.fit_transform(df)
    assert list(df.columns) == ["A", "B", "C"]
    assert list(out.columns) == ["C", "A"]
    assert list(out["A"]) == [1, 2]
    assert list(out["C"]) == ["x", "y"]


def test_select_missing_cols(df):
    selector = SelectCols(["X", "A"])
    with pytest.raises(ValueError, match="not found"):
        selector.fit(df)
    df_subset = SelectCols(["C", "A"]).fit_transform(df)
    selector = SelectCols(["A", "B"]).fit(df)
    with pytest.raises(ValueError, match="not found"):
        selector.transform(df_subset)


def test_drop_cols(df):
    selector = DropCols(["C", "A"])
    out = selector.fit_transform(df)
    assert list(df.columns) == ["A", "B", "C"]
    assert list(out.columns) == ["B"]
    assert list(out["B"]) == [10, 20]


def test_drop_missing_cols(df):
    selector = DropCols(["X", "A"])
    with pytest.raises(ValueError, match="not found"):
        selector.fit(df)
    df_subset = DropCols(["A"]).fit_transform(df)
    selector = DropCols(["A", "B"]).fit(df)
    with pytest.raises(ValueError, match="not found"):
        selector.transform(df_subset)
