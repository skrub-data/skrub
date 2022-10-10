import numpy as np
import pandas as pd
import pytest

from dirty_cat import fuzzy_join


@pytest.mark.parametrize(
    "analyzer, how",
    [("char", "left"), ("char_wb", "right"), ("word", "all")],
)
def test_fuzzy_join(analyzer, how):
    """Testing if fuzzy_join results are as expected."""

    df1 = pd.DataFrame({"a1": ["ana", "lala", "nana et sana", np.NaN]})
    df2 = pd.DataFrame({"a2": ["anna", "lala et nana", "lana", "sana", np.NaN]})

    df_joined = fuzzy_join(
        left=df1,
        right=df2,
        left_on="a1",
        right_on="a2",
        match_score=0.5,
        return_score=True,
        analyzer=analyzer,
    )

    n_cols = df1.shape[1] + df2.shape[1] + 1

    assert df_joined.shape == (len(df1.dropna()), n_cols)

    df_joined2 = fuzzy_join(
        df2,
        df1,
        left_on="a2",
        right_on="a1",
        match_score=0.45,
        return_score=True,
        analyzer=analyzer,
    )
    # Joining is always done on the left table and thus takes it shape:
    assert df_joined2.shape == (len(df2.dropna()), n_cols)

    df_joined3 = fuzzy_join(
        df2,
        df1,
        left_on="a2",
        right_on="a1",
        match_score=0.45,
        return_score=True,
        analyzer=analyzer,
    )
    pd.testing.assert_frame_equal(df_joined2, df_joined3)

    df1["a2"] = 1

    df_how = fuzzy_join(
        df1,
        df2,
        left_on="a1",
        right_on="a2",
        analyzer=analyzer,
        how=how,
        suffixes=("", "r"),
    )

    if how == "left":
        pd.testing.assert_frame_equal(df_how, df1.dropna())
    if how == "right":
        assert df_how.shape == (len(df1.dropna()), df1.shape[1])

    df_on = fuzzy_join(df_joined, df1, on="a1", analyzer=analyzer, suffixes=("1", "2"))
    assert ("a11" and "a12") in df_on.columns

    df2["a1"] = 1

    with pytest.raises(
        ValueError,
        match=r"Columns overlap but no suffix",
    ):
        fuzzy_join(df1, df2, left_on="a1", right_on="a2", suffixes=("", ""))

    if analyzer == "word":
        with pytest.warns(
            UserWarning,
            match=r"Column names overlaps.",
        ):
            fuzzy_join(
                df1, df2, left_on="a1", right_on="a2", suffixes=("a", "a"), how=how
            )

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

    df_dropna = fuzzy_join(
        left=df1,
        right=df2,
        left_on="a1",
        right_on="a2",
        analyzer=analyzer,
        match_score=0.5,
        return_score=True,
        drop_unmatched=True,
    )
    assert df_dropna.shape[0] == df_joined.dropna().shape[0]


def test_fuzzy_join_dtypes():
    # Test that the dtypes of dataframes are maintained after join
    a = pd.DataFrame({'col1': ['aaa', 'bbb'], 'col2': [1, 2]})
    b = pd.DataFrame({'col1': ['aaa_', 'bbb_'], 'col3': [1, 2]})
    c = fuzzy_join(a, b, on='col1')
    assert a.dtypes['col2'].kind == 'i'
    assert c.dtypes['col2'] == a.dtypes['col2']
    assert c.dtypes['col3'] == b.dtypes['col3']


@pytest.mark.parametrize(
    "analyzer, how, suffixes, on",
    [("a_blabla", "k_blabla", ["a", "b", "c"], ["a"]), (1, 34, [1, 2, 3], 3)],
)
def test_parameters_error(analyzer, how, suffixes, on):
    """Testing if correct errors are raised when wrong parameter values are given."""
    df1 = pd.DataFrame({"a": ["ana", "lala", "nana"], "b": [1, 2, 3]})
    df2 = pd.DataFrame({"a": ["anna", "lala", "ana", "sana"], "c": [5, 6, 7, 8]})
    with pytest.raises(
        ValueError,
        match=(
            f"analyzer should be either 'char', 'word' or 'char_wb', got {analyzer!r}"
        ),
    ):
        fuzzy_join(df1, df2, on="a", analyzer=analyzer)
    with pytest.raises(
        ValueError,
        match=f"how should be either 'left', 'right' or 'all', got {how!r}",
    ):
        fuzzy_join(df1, df2, on="a", how=how)
    with pytest.raises(
        ValueError, match="Invalid number of suffixes: expected 2, got 3"
    ):
        fuzzy_join(df1, df2, on="a", suffixes=suffixes)
    with pytest.raises(
        ValueError,
        match=(
            "Parameter 'left_on', 'right_on' or 'on' has invalid type, expected string"
        ),
    ):
        fuzzy_join(df1, df2, on=on)


def test_duplicate_columns_drop():
    df1 = pd.DataFrame(
        {"a1": ["ana", "lala", "nana et sana", np.NaN], "b": [14, 13, 51, np.NaN]}
    )
    df2 = pd.DataFrame(
        {
            "a2": ["anna", "lala et nana", "lana", "sana", np.NaN],
            "b": [1, 3, 6, 7, np.NaN],
        }
    )

    for h in ["left", "right"]:
        df_joined = fuzzy_join(
            left=df1,
            right=df2,
            left_on="a1",
            right_on="a2",
            analyzer="char",
            suffixes=("1", "1"),
            how=h,
        )
        pd.testing.assert_frame_equal(df_joined, df_joined.dropna())
