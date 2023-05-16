from typing import Literal

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import HashingVectorizer

from skrub import fuzzy_join


@pytest.mark.parametrize(
    "analyzer",
    ["char", "char_wb", "word"],
)
def test_fuzzy_join(analyzer: Literal["char", "char_wb", "word"]) -> None:
    """
    Testing if fuzzy_join results are as expected.
    """

    df1 = pd.DataFrame({"a1": ["ana", "lala", "nana et sana", np.NaN]})
    df2 = pd.DataFrame({"a2": ["anna", "lala et nana", "lana", "sana", np.NaN]})

    df_joined = fuzzy_join(
        left=df1,
        right=df2,
        left_on="a1",
        right_on="a2",
        match_score=0.45,
        return_score=True,
        analyzer=analyzer,
    )

    n_cols = df1.shape[1] + df2.shape[1] + 1

    assert df_joined.shape == (len(df1), n_cols)

    df_joined2 = fuzzy_join(
        df2,
        df1,
        how="left",
        left_on="a2",
        right_on="a1",
        match_score=0.35,
        return_score=True,
        analyzer=analyzer,
    )
    # Joining is always done on the left table and thus takes it shape:
    assert df_joined2.shape == (len(df2), n_cols)

    df_joined3 = fuzzy_join(
        df1,
        df2,
        how="right",
        right_on="a2",
        left_on="a1",
        match_score=0.35,
        return_score=True,
        analyzer=analyzer,
    )
    # We sort the index as the column order is not important here
    assert df_joined2.sort_index(axis=1).equals(df_joined3.sort_index(axis=1))

    df1["a2"] = 1

    df_on = fuzzy_join(df_joined, df1, on="a1", analyzer=analyzer, suffixes=("1", "2"))
    assert ("a11" and "a12") in df_on.columns

    df2["a1"] = 1

    df = fuzzy_join(
        df1,
        df2,
        left_on="a1",
        right_on="a2",
        analyzer=analyzer,
        return_score=True,
        suffixes=("l", "r"),
    )
    assert ("a1l" and "a1r") in df.columns


def test_fuzzy_join_dtypes() -> None:
    """
    Test that the dtypes of dataframes are maintained after join
    """
    a = pd.DataFrame({"col1": ["aaa", "bbb"], "col2": [1, 2]})
    b = pd.DataFrame({"col1": ["aaa_", "bbb_"], "col3": [1, 2]})
    c = fuzzy_join(a, b, on="col1")
    assert a.dtypes["col2"].kind == "i"
    assert c.dtypes["col2"] == a.dtypes["col2"]
    assert c.dtypes["col3"] == b.dtypes["col3"]


@pytest.mark.parametrize(
    ["analyzer", "on", "how"],
    [
        ("a_blabla", True, "left"),
        (1, 3, "right"),
    ],
)
def test_parameters_error(analyzer, on, how) -> None:
    """
    Testing if correct errors are raised when wrong parameter values are given.
    """
    df1 = pd.DataFrame({"a": ["ana", "lala", "nana"], "b": [1, 2, 3]})
    df2 = pd.DataFrame({"a": ["anna", "lala", "ana", "sana"], "c": [5, 6, 7, 8]})
    with pytest.raises(
        ValueError,
        match=(
            f"analyzer should be either 'char', 'word' or 'char_wb', got {analyzer!r}"
        ),
    ):
        fuzzy_join(df1, df2, on="a", analyzer=analyzer, how=how)
    with pytest.raises(
        TypeError,
        match=r"invalid type",
    ):
        fuzzy_join(df1, df2, on=on, how=how)
    with pytest.raises(
        TypeError,
        match=r"invalid type",
    ):
        fuzzy_join(df1, df2, on="a", match_score="blabla")
    with pytest.raises(
        ValueError,
        match=r"'numerical_match' should be either",
    ):
        fuzzy_join(df1, df2, on="a", numerical_match="wrong_name")


def test_missing_keys() -> None:
    a = pd.DataFrame({"col1": ["aaa", "bbb"], "col2": [1, 2]})
    b = pd.DataFrame({"col1": ["aaa_", "bbb_"], "col3": [1, 2]})
    with pytest.raises(
        KeyError,
        match=r"Required parameter missing",
    ):
        fuzzy_join(a, b, left_on="col1")
    left = pd.DataFrame({"a": ["aa", np.NaN, "bb"], "b": [1, 2, np.NaN]})
    right = pd.DataFrame(
        {"a": ["aa", "bb", np.NaN, "cc", "dd"], "c": [5, 6, 7, 8, np.NaN]}
    )
    output = fuzzy_join(left, right, on="a")
    assert output.shape == (3, 4)


def test_drop_unmatched() -> None:
    a = pd.DataFrame({"col1": ["aaaa", "bbb", "ddd dd"], "col2": [1, 2, 3]})
    b = pd.DataFrame({"col1": ["aaa_", "bbb_", "cc ccc"], "col3": [1, 2, 3]})

    c1 = fuzzy_join(a, b, on="col1", match_score=0.6, drop_unmatched=True)
    assert c1.shape == (2, 4)

    c2 = fuzzy_join(a, b, on="col1", match_score=0.6)
    assert sum(c2["col3"].isna()) > 0

    c3 = fuzzy_join(a, b, on="col1", how="right", match_score=0.6)
    assert sum(c3["col3"].isna()) > 0

    c4 = fuzzy_join(a, b, on="col1", how="right", match_score=0.6, drop_unmatched=True)
    assert c4.shape == (2, 4)


def test_how_param() -> None:
    """
    Test correct shape of left and right joins.
    Also test if an error is raised when an incorrect parameter value is passed.
    """
    a = pd.DataFrame({"col1": ["aaaa", "bbb", "ddd dd"], "col2": [1, 2, 3]})
    b = pd.DataFrame(
        {
            "col2": ["aaa_", "bbb_", "ccc", "ddd", "eee", "fff"],
            "col3": [1, 2, 3, 4, 5, 6],
        }
    )

    c = fuzzy_join(a, b, left_on="col1", right_on="col2", how="left")
    assert c.shape == (len(a), 4)

    c = fuzzy_join(a, b, left_on="col1", right_on="col2", how="right")
    assert c.shape == (len(b), 4)

    with pytest.raises(
        ValueError,
        match=r"Parameter 'how' should be either ",
    ):
        c = fuzzy_join(a, b, how="inner")


def test_fuzzy_join_pandas_comparison() -> None:
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
            "key": ["K0", "K1", "K2", "K3"],
            "C": ["C0", "C1", "C2", "C3"],
            "D": ["D0", "D1", "D2", "D3"],
        }
    )

    result = pd.merge(left, right, on="key", how="left")
    result_fj = fuzzy_join(left, right, on="key", how="left")

    result_fj.drop(columns=["key_y"], inplace=True)
    result_fj.rename(columns={"key_x": "key"}, inplace=True)

    pd.testing.assert_frame_equal(result, result_fj)

    result_r = pd.merge(left, right, on="key", how="right")
    result_r_fj = fuzzy_join(left, right, on="key", how="right")

    result_r_fj.drop(columns=["key_y"], inplace=True)
    result_r_fj.rename(columns={"key_x": "key"}, inplace=True)

    pd.testing.assert_frame_equal(result_r, result_r_fj)

    left = left.sample(frac=1, random_state=0)
    right = right.sample(frac=1, random_state=0)
    result_s = pd.merge(left, right, on="key", how="left", sort=True)
    result_s_fj = fuzzy_join(left, right, on="key", how="left", sort=True)

    result_s_fj.drop(columns=["key_y"], inplace=True)
    result_s_fj.rename(columns={"key_x": "key"}, inplace=True)

    pd.testing.assert_frame_equal(result_s, result_s_fj)


def test_correct_encoder() -> None:
    """
    Test that the encoder error checking is working as intended.
    """

    class TestVectorizer(HashingVectorizer):
        """
        Implements a custom vectorizer to check if the `encoder`
        parameter uses the passed instance as expected.
        Raises an error when `fit` is called.
        """

        def fit(self, X):
            raise AssertionError("Custom vectorizer was called as intended.")

    left = pd.DataFrame(
        {
            "key": ["K0", "K1", "K2", "K3"],
            "A": ["A0", "A1", "A2", "A3"],
            "B": ["B0", "B1", "B2", "B3"],
        }
    )

    right = pd.DataFrame(
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
        fuzzy_join(left, right, on="key", how="left", encoder=enc)

    with pytest.raises(
        ValueError, match=r"Parameter 'encoder' should be a vectorizer "
    ):
        fuzzy_join(left, right, on="key", how="left", encoder="awrongencoder")


def test_numerical_column() -> None:
    """
    Testing that fuzzy_join works with numerical columns.
    """

    left = pd.DataFrame({"str1": ["aa", "a", "bb"], "int": [10, 2, 5]})
    right = pd.DataFrame(
        {
            "str2": ["aa", "bb", "a", "cc", "dd"],
            "int": [55, 6, 2, 15, 6],
        }
    )

    fj_num = fuzzy_join(left, right, on="int", numerical_match="number")
    n_cols = left.shape[1] + right.shape[1]

    assert fj_num.shape == (len(left), n_cols)

    fj_num2 = fuzzy_join(
        left, right, on="int", numerical_match="number", return_score=True
    )
    assert fj_num2.shape == (len(left), n_cols + 1)

    fj_num3 = fuzzy_join(
        left,
        right,
        on="int",
        numerical_match="number",
        match_score=0.8,
        drop_unmatched=True,
    )
    assert fj_num3.shape == (2, n_cols)


def test_multiple_keys() -> None:
    """
    Test fuzzy joining on multiple keys with possibly mixed types.
    """

    left = pd.DataFrame(
        {
            "str1": ["Paris", "Paris", "Paris"],
            "str2": ["Texas", "France", "Greek God"],
            "int1": [10, 2, 5],
            "int2": [103, 250, 532],
        }
    )
    right = pd.DataFrame(
        {
            "str_1": ["Paris", "Paris", "Paris", "cc", "dd"],
            "str_2": ["TX", "FR", "GR Mytho", "cc", "dd"],
            "int1": [55, 6, 2, 15, 6],
            "int2": [554, 146, 32, 215, 612],
        }
    )

    # On multiple numeric keys
    fj_num = fuzzy_join(left, right, on=["int1", "int2"], numerical_match="number")
    assert fj_num.shape == (3, 8)

    # On multiple string keys
    fj_str = fuzzy_join(
        left, right, left_on=["str1", "str2"], right_on=["str_1", "str_2"]
    )
    assert fj_str.shape == (3, 8)

    # On mixed, numeric and string keys
    fj_mixed = fuzzy_join(
        left,
        right,
        left_on=["str1", "str2", "int2"],
        right_on=["str_1", "str_2", "int2"],
        numerical_match="number",
    )
    assert fj_mixed.shape == (3, 8)


def test_iterable_input() -> None:
    """
    Test if iterable input: list, set, dictionary or tuple works.
    """
    df1 = pd.DataFrame(
        {"a": ["ana", "lala", "nana"], "str2": ["Texas", "France", "Greek God"]}
    )
    df2 = pd.DataFrame(
        {"a": ["anna", "lala", "ana", "nnana"], "str_2": ["TX", "FR", "GR Mytho", "dd"]}
    )
    assert fuzzy_join(df1, df2, on=["a"]).shape == (3, 4)
    assert fuzzy_join(df1, df2, on={"a"}).shape == (3, 4)
    assert fuzzy_join(df1, df2, on="a").shape == (3, 4)

    assert fuzzy_join(
        df1, df2, left_on=["a", "str2"], right_on={"a", "str_2"}
    ).shape == (3, 4)
    assert fuzzy_join(
        df1, df2, left_on=("a", "str2"), right_on={"a", "str_2"}
    ).shape == (3, 4)


def test_missing_values() -> None:
    """
    Test fuzzy joining on missing values.
    """
    a = pd.DataFrame({"col1": ["aaaa", "bbb", "ddd dd"], "col2": [1, 2, 3]})
    b = pd.DataFrame({"col3": [np.NaN, "bbb", "ddd dd"], "col4": [1, 2, 3]})

    with pytest.warns(UserWarning, match=r"merging on missing values"):
        c = fuzzy_join(a, b, left_on="col1", right_on="col3", how="right")
    assert c.shape[0] == len(b)

    with pytest.warns(UserWarning, match=r"merging on missing values"):
        c = fuzzy_join(b, a, left_on="col3", right_on="col1", return_score=True)
    assert c.shape[0] == len(b)
