import pytest

from skrub._dispatch import dispatch


def test_dispatch():
    @dispatch
    def f(x, y=None):
        return "default"

    @f.specialize("pandas")
    def _(x, y=None):
        return "pandas"

    @f.specialize("pandas", argument_type="Column")
    def _(x, y=None):
        return "pandas series"

    @f.specialize("polars")
    def _(x, y=None):
        return "polars"

    assert f(0) == "default"

    import pandas as pd

    df = pd.DataFrame(dict(a=[1, 2, 3]))
    assert f(df) == "pandas"
    assert f(0, df) == "default"
    assert f(df["a"]) == "pandas series"

    try:
        import polars as pl
    except ImportError:
        pytest.skip("polars not installed")

    df = pl.DataFrame(dict(a=[1, 2, 3]))
    assert f(df) == "polars"
    assert f(df["a"]) == "polars"
    assert f(0, df["a"]) == "default"

    with pytest.raises(KeyError, match="Unknown dataframe module"):
        f.specialize("numpy")
