import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from skrub import _dataframe as sbd
from skrub._clean_categories import CleanCategories
from skrub._on_each_column import RejectColumn


def test_clean_categories_polars():
    # polars categorical columns are passed through, non-categorical are
    # rejected during fit but converted to categorical during transform.
    pl = pytest.importorskip("polars")
    s = pl.Series(["a", "b", None], dtype=pl.Categorical)
    assert CleanCategories().fit_transform(s) is s
    assert CleanCategories().fit(s).transform(s) is s
    s = pl.Series(["a", "b", None], dtype=pl.Enum(["a", "b", "c"]))
    assert CleanCategories().fit_transform(s) is s
    assert CleanCategories().fit(s).transform(s) is s


def test_clean_categories_pandas():
    s = pd.Series(["a", "b", None], dtype="string").astype("category")
    expected = pd.Series(["a", "b", None], dtype="category")
    assert_series_equal(CleanCategories().fit_transform(s), expected)

    s = pd.Series(["a", 2.2, None]).astype("category")
    expected = pd.Series(["a", 2.2, None], dtype="str").astype("category")
    assert_series_equal(CleanCategories().fit_transform(s), expected)


def test_collapsing_categories():
    class A:
        def __repr__(self):
            return "A"

    s = pd.Series([A(), A()], dtype="category")
    assert len(s.cat.categories) == 2
    out = CleanCategories().fit_transform(s)
    assert len(out.cat.categories) == 1


def test_clean_categories(df_module):
    with pytest.raises(RejectColumn):
        CleanCategories().fit_transform(df_module.make_column("c", ["a", "b"]))
    s = sbd.to_categorical(df_module.make_column("c", ["a", "b"]))
    cleaner = CleanCategories().fit(s)
    for vals in ["x", "y", None], [1.1, 2.2, None]:
        assert sbd.is_categorical(cleaner.transform(df_module.make_column("c", vals)))
