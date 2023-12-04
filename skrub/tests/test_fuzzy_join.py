import warnings
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.feature_extraction.text import HashingVectorizer

from skrub import fuzzy_join
from skrub._dataframe._polars import POLARS_SETUP
from skrub._dataframe._test_utils import is_module_polars

MODULES = [pd]
ASSERT_TUPLES = [(pd, assert_frame_equal)]

if POLARS_SETUP:
    import polars as pl
    from polars.testing import assert_frame_equal as assert_frame_equal_pl

    MODULES.append(pl)
    ASSERT_TUPLES.append((pl, assert_frame_equal_pl))


@pytest.mark.parametrize("px", MODULES)
@pytest.mark.parametrize(
    "analyzer",
    ["char", "char_wb", "word"],
)
def test_fuzzy_join(px, analyzer: Literal["char", "char_wb", "word"]):
    """
    Testing if fuzzy_join results are as expected.
    """
    if is_module_polars(px):
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'")
    df1 = px.DataFrame({"a1": ["ana", "lala", "nana et sana", np.NaN]})
    df2 = px.DataFrame({"a2": ["anna", "lala et nana", "lana", "sana", np.NaN]})

    df_joined = fuzzy_join(
        left=df1,
        right=df2,
        left_on="a1",
        right_on="a2",
        max_dist=1.0,
        add_match_info=True,
    )

    n_cols = df1.shape[1] + df2.shape[1] + 3

    assert df_joined.shape == (len(df1), n_cols)

    df_joined2 = fuzzy_join(
        df2,
        df1,
        left_on="a2",
        right_on="a1",
        max_dist=1.0,
        add_match_info=True,
    )
    # Joining is always done on the left table and thus takes it shape:
    assert df_joined2.shape == (len(df2), n_cols)

    df1["a2"] = 1

    df_on = fuzzy_join(df_joined, df1, on="a1", suffix="2")
    assert "a12" in df_on.columns


@pytest.mark.parametrize("px", MODULES)
def test_max_dist(px):
    if is_module_polars(px):
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'")
    left = px.DataFrame({"A": ["aa", "bb"]})
    right = px.DataFrame({"A": ["aa", "ba"], "B": [1, 2]})
    join = fuzzy_join(left, right, on="A", suffix="r")
    assert join["Br"].to_list() == [1, 2]
    join = fuzzy_join(left, right, on="A", suffix="r", max_dist=0.5)
    assert join["Br"].fillna(-1).to_list() == [1, -1]


@pytest.mark.parametrize("px", MODULES)
def test_perfect_matches(px):
    if is_module_polars(px):
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'")
    # non-regression test for https://github.com/skrub-data/skrub/issues/764
    # fuzzy_join when all rows had a perfect match used to trigger a division by 0
    df = px.DataFrame({"A": [0, 1]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", message="This feature is still experimental")
        join = fuzzy_join(df, df, on="A", suffix="r", add_match_info=True)
    assert_array_equal(join["skrub_Joiner_rescaled_distance"].to_numpy(), [0.0, 0.0])


@pytest.mark.parametrize("px", MODULES)
def test_fuzzy_join_dtypes(px):
    """
    Test that the dtypes of dataframes are maintained after join
    """
    if is_module_polars(px):
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'")
    a = px.DataFrame({"col1": ["aaa", "bbb"], "col2": [1, 2]})
    b = px.DataFrame({"col1": ["aaa_", "bbb_"], "col3": [1, 2]})
    c = fuzzy_join(a, b, on="col1", suffix="r")
    assert a.dtypes["col2"].kind == "i"
    assert c.dtypes["col2"] == a.dtypes["col2"]
    assert c.dtypes["col3r"] == b.dtypes["col3"]


@pytest.mark.parametrize("px", MODULES)
def test_missing_keys(px):
    if is_module_polars(px):
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'")
    a = px.DataFrame({"col1": ["aaa", "bbb"], "col2": [1, 2]})
    b = px.DataFrame({"col1": ["aaa_", "bbb_"], "col3": [1, 2]})
    with pytest.raises(
        ValueError,
        match=r"Must pass",
    ):
        fuzzy_join(a, b, left_on="col1", suffix="r", add_match_info=False)
    left = px.DataFrame({"a": ["aa", np.NaN, "bb"], "b": [1, 2, np.NaN]})
    right = px.DataFrame(
        {"a": ["aa", "bb", np.NaN, "cc", "dd"], "c": [5, 6, 7, 8, np.NaN]}
    )
    output = fuzzy_join(left, right, on="a", suffix="r", add_match_info=False)
    assert output.shape == (3, 4)


@pytest.mark.parametrize("px", MODULES)
def test_drop_unmatched(px):
    if is_module_polars(px):
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'")
    a = px.DataFrame({"col1": ["aaaa", "bbb", "ddd dd"], "col2": [1, 2, 3]})
    b = px.DataFrame({"col1": ["aaa_", "bbb_", "cc ccc"], "col3": [1, 2, 3]})

    c1 = fuzzy_join(
        a,
        b,
        on="col1",
        max_dist=0.9,
        drop_unmatched=True,
        suffix="r",
        add_match_info=False,
    )
    assert c1.shape == (2, 4)
    c2 = fuzzy_join(a, b, on="col1", max_dist=0.9, suffix="r", add_match_info=False)
    assert sum(c2["col3r"].isna()) > 0


def test_fuzzy_join_pandas_comparison():
    """
    Tests if fuzzy_join's output is as similar as
    possible with `pandas.merge`.
    """
    left = pd.DataFrame(
        {
            "key": ["K0", "K1", "K2", "K3"],
            "A": ["A0", "A1", "A2", "A3"],
            "B": ["B0", "B1", "B2", "B3"],
        }
    )

    right = pd.DataFrame(
        {
            "key_": ["K0", "K1", "K2", "K3"],
            "C": ["C0", "C1", "C2", "C3"],
            "D": ["D0", "D1", "D2", "D3"],
        }
    )

    result = pd.merge(left, right, left_on="key", right_on="key_")
    result_fj = fuzzy_join(
        left, right, left_on="key", right_on="key_", add_match_info=False
    )

    assert_frame_equal(result, result_fj)


@pytest.mark.parametrize("px", MODULES)
def test_correct_encoder(px):
    """
    Test that the encoder error checking is working as intended.
    """
    if is_module_polars(px):
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'")

    class TestVectorizer(HashingVectorizer):
        """
        Implements a custom vectorizer to check if the `encoder`
        parameter uses the passed instance as expected.
        Raises an error when `fit` is called.
        """

        def fit(self, X, y=None):
            raise AssertionError("Custom vectorizer was called as intended.")

    left = px.DataFrame(
        {
            "key": ["K0", "K1", "K2", "K3"],
            "A": ["A0", "A1", "A2", "A3"],
            "B": ["B0", "B1", "B2", "B3"],
        }
    )

    right = px.DataFrame(
        {
            "key": ["K0", "K1", "K2", "K3"],
            "C": ["C0", "C1", "C2", "C3"],
            "D": ["D0", "D1", "D2", "D3"],
        }
    )

    enc = TestVectorizer()

    with pytest.raises(
        AssertionError, match=r"Custom vectorizer was called as intended."
    ):
        fuzzy_join(
            left, right, on="key", suffix="_", string_encoder=enc, add_match_info=False
        )


@pytest.mark.parametrize("px", MODULES)
def test_numerical_column(px):
    """
    Testing that fuzzy_join works with numerical columns.
    """
    if is_module_polars(px):
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'")
    left = px.DataFrame({"str1": ["aa", "a", "bb"], "int": [10, 2, 5]})
    right = px.DataFrame(
        {
            "str2": ["aa", "bb", "a", "cc", "dd"],
            "int": [55, 6, 2, 15, 6],
        }
    )

    fj_num = fuzzy_join(left, right, on="int", suffix="r", add_match_info=False)
    n_cols = left.shape[1] + right.shape[1]
    n_samples = len(left)

    assert fj_num.shape == (n_samples, n_cols)

    fj_num2 = fuzzy_join(left, right, on="int", add_match_info=True, suffix="r")
    assert fj_num2.shape == (n_samples, n_cols + 3)

    fj_num3 = fuzzy_join(
        left,
        right,
        on="int",
        max_dist=3,
        drop_unmatched=True,
        suffix="r",
        ref_dist="no_rescaling",
        add_match_info=False,
    )
    assert fj_num3.shape == (2, n_cols)


@pytest.mark.parametrize("px, assert_frame_equal_", ASSERT_TUPLES)
def test_datetime_column(px, assert_frame_equal_):
    """
    Testing that fuzzy_join works with datetime columns.
    """
    if is_module_polars(px):
        pytest.xfail(reason="Module 'polars' has no attribute 'to_datetime'")
    left = px.DataFrame(
        {
            "str1": ["aa", "a", "bb"],
            "date": px.to_datetime(["10/10/2022", "12/11/2021", "09/25/2011"]),
        }
    )
    right = px.DataFrame(
        {
            "str2": ["aa", "bb", "a", "cc", "dd"],
            "date": px.to_datetime(
                ["09/10/2022", "12/24/2021", "09/25/2010", "11/05/2025", "02/21/2000"]
            ),
        }
    )

    fj_time = fuzzy_join(left, right, on="date", suffix="r", add_match_info=False)

    fj_time_expected = px.DataFrame(
        {
            "str1": ["aa", "a", "bb"],
            "date": px.to_datetime(["10/10/2022", "12/11/2021", "09/25/2011"]),
            "str2r": ["aa", "bb", "a"],
            "dater": px.to_datetime(["09/10/2022", "12/24/2021", "09/25/2010"]),
        }
    )
    assert_frame_equal_(fj_time, fj_time_expected)

    n_cols = left.shape[1] + right.shape[1]
    n_samples = len(left)

    assert fj_time.shape == (n_samples, n_cols)

    fj_time2 = fuzzy_join(left, right, on="date", add_match_info=True, suffix="r")
    assert fj_time2.shape == (n_samples, n_cols + 3)

    fj_time3 = fuzzy_join(
        left,
        right,
        on="date",
        max_dist=0.1,
        drop_unmatched=True,
        suffix="r",
        add_match_info=False,
    )
    assert fj_time3.shape == (2, n_cols)


@pytest.mark.parametrize("px, assert_frame_equal_", ASSERT_TUPLES)
def test_mixed_joins(px, assert_frame_equal_):
    """
    Test fuzzy joining on mixed and multiple column types.
    """
    if is_module_polars(px):
        pytest.xfail(reason="Module 'polars' has no attribute 'to_datetime'")
    left = px.DataFrame(
        {
            "str1": ["Paris", "Paris", "Paris"],
            "str2": ["Texas", "France", "Greek God"],
            "int1": [10, 2, 5],
            "int2": [103, 250, 532],
            "date": px.to_datetime(["10/10/2022", "12/11/2021", "09/25/2011"]),
        }
    )
    right = px.DataFrame(
        {
            "str_1": ["Paris", "Paris", "Paris", "cc", "dd"],
            "str_2": ["TX", "FR", "GR Mytho", "cc", "dd"],
            "int1": [55, 6, 2, 15, 6],
            "int2": [554, 146, 32, 215, 612],
            "date": px.to_datetime(
                ["09/10/2022", "12/24/2021", "09/25/2010", "11/05/2025", "02/21/2000"]
            ),
        }
    )

    # On multiple numeric keys
    fj_num = fuzzy_join(
        left, right, on=["int1", "int2"], suffix="r", add_match_info=False
    )

    expected_fj_num = px.DataFrame(
        {
            "str1": ["Paris", "Paris", "Paris"],
            "str2": ["Texas", "France", "Greek God"],
            "int1": [10, 2, 5],
            "int2": [103, 250, 532],
            "date": px.to_datetime(["10/10/2022", "12/11/2021", "09/25/2011"]),
            "str_1r": ["Paris", "Paris", "dd"],
            "str_2r": ["FR", "FR", "dd"],
            "int1r": [6, 6, 6],
            "int2r": [146, 146, 612],
            "dater": px.to_datetime(["12/24/2021", "12/24/2021", "02/21/2000"]),
        }
    )
    assert_frame_equal_(fj_num, expected_fj_num)
    assert fj_num.shape == (3, 10)

    # On multiple string keys
    fj_str = fuzzy_join(
        left,
        right,
        left_on=["str1", "str2"],
        right_on=["str_1", "str_2"],
        suffix="r",
        add_match_info=False,
    )

    expected_fj_str = px.DataFrame(
        {
            "str1": ["Paris", "Paris", "Paris"],
            "str2": ["Texas", "France", "Greek God"],
            "int1": [10, 2, 5],
            "int2": [103, 250, 532],
            "date": px.to_datetime(["2022-10-10", "2021-12-11", "2011-09-25"]),
            "str_1r": ["Paris", "Paris", "Paris"],
            "str_2r": ["TX", "FR", "GR Mytho"],
            "int1r": [55, 6, 2],
            "int2r": [554, 146, 32],
            "dater": px.to_datetime(["2022-09-10", "2021-12-24", "2010-09-25"]),
        }
    )
    assert_frame_equal_(fj_str, expected_fj_str)
    assert fj_str.shape == (3, 10)

    # On mixed, numeric and string keys
    fj_mixed = fuzzy_join(
        left,
        right,
        left_on=["str1", "str2", "int2"],
        right_on=["str_1", "str_2", "int2"],
        suffix="r",
        add_match_info=False,
    )
    expected_fj_mixed = px.DataFrame(
        {
            "str1": ["Paris", "Paris", "Paris"],
            "str2": ["Texas", "France", "Greek God"],
            "int1": [10, 2, 5],
            "int2": [103, 250, 532],
            "date": px.to_datetime(["2022-10-10", "2021-12-11", "2011-09-25"]),
            "str_1r": ["Paris", "Paris", "Paris"],
            "str_2r": ["FR", "FR", "TX"],
            "int1r": [6, 6, 55],
            "int2r": [146, 146, 554],
            "dater": px.to_datetime(["2021-12-24", "2021-12-24", "2022-09-10"]),
        }
    )
    assert_frame_equal_(fj_mixed, expected_fj_mixed)
    assert fj_mixed.shape == (3, 10)

    # On mixed time and string keys
    fj_mixed2 = fuzzy_join(
        left,
        right,
        left_on=["str1", "date"],
        right_on=["str_1", "date"],
        suffix="r",
        add_match_info=False,
    )
    expected_fj_mixed2 = px.DataFrame(
        {
            "str1": ["Paris", "Paris", "Paris"],
            "str2": ["Texas", "France", "Greek God"],
            "int1": [10, 2, 5],
            "int2": [103, 250, 532],
            "date": px.to_datetime(["2022-10-10", "2021-12-11", "2011-09-25"]),
            "str_1r": ["Paris", "Paris", "Paris"],
            "str_2r": ["TX", "FR", "GR Mytho"],
            "int1r": [55, 6, 2],
            "int2r": [554, 146, 32],
            "dater": px.to_datetime(["2022-09-10", "2021-12-24", "2010-09-25"]),
        }
    )
    assert_frame_equal_(fj_mixed2, expected_fj_mixed2)
    assert fj_mixed2.shape == (3, 10)

    # On mixed time and numbers keys
    fj_mixed3 = fuzzy_join(
        left,
        right,
        left_on=["int1", "date"],
        right_on=["int1", "date"],
        suffix="r",
        add_match_info=False,
    )
    expected_fj_mixed3 = px.DataFrame(
        {
            "str1": ["Paris", "Paris", "Paris"],
            "str2": ["Texas", "France", "Greek God"],
            "int1": [10, 2, 5],
            "int2": [103, 250, 532],
            "date": px.to_datetime(["2022-10-10", "2021-12-11", "2011-09-25"]),
            "str_1r": ["Paris", "Paris", "Paris"],
            "str_2r": ["FR", "FR", "GR Mytho"],
            "int1r": [6, 6, 2],
            "int2r": [146, 146, 32],
            "dater": px.to_datetime(["2021-12-24", "2021-12-24", "2010-09-25"]),
        }
    )
    assert_frame_equal_(fj_mixed3, expected_fj_mixed3)
    assert fj_mixed3.shape == (3, 10)


@pytest.mark.parametrize("px", MODULES)
def test_iterable_input(px):
    """
    Test if iterable input: list, set, dictionary or tuple works.
    """
    if is_module_polars(px):
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'")
    df1 = px.DataFrame(
        {"a": ["ana", "lala", "nana"], "str2": ["Texas", "France", "Greek God"]}
    )
    df2 = px.DataFrame(
        {
            "a": ["anna", "lala", "ana", "nnana"],
            "str_2": ["TX", "FR", "GR Mytho", "dd"],
        }
    )
    assert fuzzy_join(df1, df2, on=["a"], suffix="r", add_match_info=False).shape == (
        3,
        4,
    )
    assert fuzzy_join(df1, df2, on="a", suffix="r", add_match_info=False).shape == (
        3,
        4,
    )

    assert fuzzy_join(
        df1,
        df2,
        left_on=["a", "str2"],
        right_on=("a", "str_2"),
        suffix="r",
        add_match_info=False,
    ).shape == (3, 4)
    assert fuzzy_join(
        df1,
        df2,
        left_on=("a", "str2"),
        right_on=np.asarray(("a", "str_2")),
        suffix="r",
        add_match_info=False,
    ).shape == (3, 4)


@pytest.mark.xfail
@pytest.mark.parametrize("px", MODULES)
def test_missing_values(px):
    """
    Test fuzzy joining on missing values.
    """
    if is_module_polars(px):
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'")
    a = px.DataFrame({"col1": ["aaaa", "bbb", "ddd dd"], "col2": [1, 2, 3]})
    b = px.DataFrame({"col3": [np.NaN, "bbb", "ddd dd"], "col4": [1, 2, 3]})

    with pytest.warns(UserWarning, match=r"merging on missing values"):
        c = fuzzy_join(a, b, left_on="col1", right_on="col3", add_match_info=False)
    assert c.shape[0] == len(b)

    with pytest.warns(UserWarning, match=r"merging on missing values"):
        c = fuzzy_join(b, a, left_on="col3", right_on="col1", add_match_info=True)
    assert c.shape[0] == len(b)
