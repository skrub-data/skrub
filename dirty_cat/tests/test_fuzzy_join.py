import numpy as np
import pandas as pd
import pytest

from dirty_cat import fuzzy_join


@pytest.mark.parametrize(
    "analyzer, how",
    [("char", "left"), ("char_wb", "right"), ("word", "left")],
)
def test_fuzzy_join(analyzer, how):
    """Testing if fuzzy_join gives joining results as expected."""

    df1 = pd.DataFrame({"a1": ["ana", "lala", "nana"]})
    df2 = pd.DataFrame({"a2": ["anna", "lala", "lana", "sana"]})
    ground_truth = pd.DataFrame(
        {"a1": ["ana", "lala", "nana"], "a2": ["anna", "lala", np.NaN]}
    )

    df_joined = fuzzy_join(
        left=df1,
        right=df2,
        left_on="a1",
        right_on="a2",
        return_score=True,
        analyzer=analyzer,
        match_score=0.6,
        how="all",
    )

    n_cols = df1.shape[1] + df2.shape[1] + 1

    assert df_joined.shape == (len(df1), n_cols)
    pd.testing.assert_frame_equal(
        df_joined.drop("matching_score", axis=1), ground_truth
    )

    df_joined2 = fuzzy_join(
        df2,
        df1,
        left_on="a2",
        right_on="a1",
        return_score=True,
        analyzer=analyzer,
        match_score=0.6,
    )
    # Joining is always done on the left table and thus takes it shape:
    assert df_joined2.shape == (df2.shape[0], n_cols)

    df_joined3 = fuzzy_join(
        df2,
        df1,
        left_on="a2",
        right_on="a1",
        return_score=True,
        analyzer=analyzer,
        match_score=0.6,
    )
    pd.testing.assert_frame_equal(df_joined2, df_joined3)

    df_how = fuzzy_join(
        df1,
        df2,
        left_on="a1",
        right_on="a2",
        analyzer=analyzer,
        match_score=0.6,
        how=how,
    )
    if how == "left":
        pd.testing.assert_frame_equal(df_how, df1)
    if how == "right":
        assert df_how.shape == df1.shape

    df_on = fuzzy_join(
        df_joined, df1, on=["a1"], analyzer=analyzer, suffixes=("1", "2")
    )
    assert "a11" and "a12" in df_on.columns


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
        match="Parameter left_on, right_on or on has invalid type, expected string",
    ):
        fuzzy_join(df1, df2, on=on)
