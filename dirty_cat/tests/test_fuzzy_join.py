import numpy as np
import pandas as pd
import pytest

from dirty_cat import fuzzy_join


@pytest.mark.parametrize("analyzer", ["char", "char_wb", "word"])
def test_fuzzy_join(analyzer):
    """Testing if fuzzy_join results are as expected."""

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

    assert df_joined.shape == (len(df1.dropna()), n_cols)

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
    assert df_joined2.shape == (len(df2.dropna()), n_cols)

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
    pd.testing.assert_frame_equal(df_joined2, df_joined3)

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


def test_fuzzy_join_dtypes():
    # Test that the dtypes of dataframes are maintained after join
    a = pd.DataFrame({"col1": ["aaa", "bbb"], "col2": [1, 2]})
    b = pd.DataFrame({"col1": ["aaa_", "bbb_"], "col3": [1, 2]})
    c = fuzzy_join(a, b, on="col1")
    assert a.dtypes["col2"].kind == "i"
    assert c.dtypes["col2"] == a.dtypes["col2"]
    assert c.dtypes["col3"] == b.dtypes["col3"]


@pytest.mark.parametrize(
    "analyzer, on, how",
    [("a_blabla", ["a"], "left"), (1, 3, "right")],
)
def test_parameters_error(analyzer, on, how):
    """Testing if correct errors are raised when wrong parameter values are given."""
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
        KeyError,
        match=r"invalid type",
    ):
        fuzzy_join(df1, df2, on=on, how=how)


def test_missing_keys():
    a = pd.DataFrame({"col1": ["aaa", "bbb"], "col2": [1, 2]})
    b = pd.DataFrame({"col1": ["aaa_", "bbb_"], "col3": [1, 2]})
    with pytest.raises(
        KeyError,
        match="Required parameter missing: either parameter",
    ):
        fuzzy_join(a, b, left_on="col1")


def test_drop_unmatched():
    a = pd.DataFrame({"col1": ["aaaa", "bbb", "ddd dd"], "col2": [1, 2, 3]})
    b = pd.DataFrame({"col1": ["aaa_", "bbb_", "cc ccc"], "col3": [1, 2, 3]})
    c = fuzzy_join(a, b, on="col1", match_score=0.5, drop_unmatched=True)
    assert c.shape == (2, 4)

    a = pd.DataFrame({"col1": ["aaaa", "bbb", "ddd dd"], "col2": [1, 2, 3]})
    b = pd.DataFrame({"col1": ["aaa_", "bbb_", "cc ccc"], "col3": [1, 2, 3]})
    c = fuzzy_join(a, b, on="col1", match_score=0.5)
    assert sum(c["col3"].isna()) > 0


def test_how_param():
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
        match=r"how should be either 'left' or 'right', got",
    ):
        c = fuzzy_join(a, b, how="inner")
