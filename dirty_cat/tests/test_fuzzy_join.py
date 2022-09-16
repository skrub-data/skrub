import pandas as pd
import pytest

from dirty_cat import fuzzy_join


@pytest.mark.parametrize(
    "analyzer, how",
    [("char", "left"), ("char_wb", "right")],
)
def test_fuzzy_join(analyzer, how):
    """Testing if fuzzy_join gives joining results as expected."""
    teams1 = pd.DataFrame(
        {
            "basketball_teams": [
                "LA Lakers",
                "Charlotte Hornets",
                "Polonia Warszawa",
                "Asseco",
                "Melbourne United (basketball)",
                "Partizan Belgrade",
                "Liaoning FL",
                "P.A.O.K. BC",
                "New Orlean Plicans",
            ]
        }
    )
    teams2 = pd.DataFrame(
        {
            "teams_basketball": [
                "Partizan BC",
                "New Orleans Pelicans",
                "Charlotte Hornets",
                "Polonia Warszawa (basketball)",
                "Real Madrid Baloncesto",
                "Los Angeles Lakers",
                "Asseco Gdynia",
                "PAOK BC",
                "Melbourne United",
                "Liaoning Flying Leopards",
            ]
        }
    )

    ground_truth = pd.DataFrame(
        {
            "basketball_teams": [
                "LA Lakers",
                "Charlotte Hornets",
                "Polonia Warszawa",
                "Asseco",
                "Melbourne United (basketball)",
                "Partizan Belgrade",
                "Liaoning FL",
                "P.A.O.K. BC",
                "New Orlean Plicans",
            ],
            "teams_basketball": [
                "Los Angeles Lakers",
                "Charlotte Hornets",
                "Polonia Warszawa (basketball)",
                "Asseco Gdynia",
                "Melbourne United",
                "Partizan BC",
                "Liaoning Flying Leopards",
                "PAOK BC",
                "New Orleans Pelicans",
            ],
        }
    )

    # Check correct shapes of outputs:
    teams_joined, dist1 = fuzzy_join(
        teams1,
        teams2,
        left_on="basketball_teams",
        right_on="teams_basketball",
        return_distance=True,
        analyzer=analyzer,
        threshold=0.1,
    )
    assert teams_joined.shape == (9, 2)
    assert dist1.shape == (9, 1)
    assert (teams_joined == ground_truth).all()[1]

    # And on the other way around:
    teams_joined_2, dist2 = fuzzy_join(
        teams2,
        teams1,
        left_on="teams_basketball",
        right_on="basketball_teams",
        return_distance=True,
        analyzer=analyzer,
        threshold=0.1,
    )
    # Joining is always done on the left table and thus takes it shape:
    assert teams_joined_2.shape == (10, 2)
    assert dist2.shape == (10, 1)

    # Check invariability of joining:
    teams_joined_3 = fuzzy_join(
        teams2,
        teams1,
        left_on="teams_basketball",
        right_on="basketball_teams",
        analyzer=analyzer,
        threshold=0.1,
    )
    pd.testing.assert_frame_equal(teams_joined_2, teams_joined_3)

    # Check how argument:
    teams_kept = fuzzy_join(
        teams1,
        teams2,
        left_on="basketball_teams",
        right_on="teams_basketball",
        analyzer=analyzer,
        threshold=0.1,
        how=how,
    )
    if how == "left":
        pd.testing.assert_frame_equal(teams_kept, teams1)
    if how == "right":
        assert teams_kept.shape == teams1.shape


@pytest.mark.parametrize(
    "analyzer, how, suffixes",
    [("a_blabla", "k_blabla", ["a", "b", "c"]), (1, 34, [1, 2, 3])],
)
def test_parameters_error(analyzer, how, suffixes):
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
            "value ['a'] was specified for parameter, which has invalid type,"
            " expected string."
        ),
    ):
        fuzzy_join(df1, df2, on=["a"])
