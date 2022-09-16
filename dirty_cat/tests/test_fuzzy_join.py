import pandas as pd
import pytest

# isort : off
from dirty_cat.experimental import enable_fuzzy_join  # noqa

from dirty_cat import fuzzy_join, print_worst_matches  # isort: skip

# isort : on


@pytest.mark.parametrize(
    "analyzer, match_type, how",
    [("char", "nearest", "left"), ("char_wb", "radius", "right")],
)
def test_fuzzy_join(analyzer, match_type, how):
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
        on=["basketball_teams", "teams_basketball"],
        return_distance=True,
        analyzer=analyzer,
        match_type=match_type,
        match_threshold=0.1,
    )
    assert teams_joined.shape == (9, 2)
    assert dist1.shape == (9, 1)
    assert (teams_joined == ground_truth).all()[1]

    wm = print_worst_matches(teams_joined, dist1, n=2)
    assert wm.shape == (2, 3)

    # And on the other way around:
    teams_joined_2, dist2 = fuzzy_join(
        teams2,
        teams1,
        on=["teams_basketball", "basketball_teams"],
        return_distance=True,
        analyzer=analyzer,
        match_type=match_type,
        match_threshold=0.1,
    )
    # Joining is always done on the left table and thus takes it shape:
    assert teams_joined_2.shape == (10, 2)
    assert dist2.shape == (10, 1)

    wm = print_worst_matches(teams_joined, dist1, n=6)
    assert wm.shape == (6, 3)

    # Check invariability of joining:
    teams_joined_3 = fuzzy_join(
        teams2,
        teams1,
        on=["teams_basketball", "basketball_teams"],
        analyzer=analyzer,
        match_type=match_type,
        match_threshold=0.1,
    )
    pd.testing.assert_frame_equal(teams_joined_2, teams_joined_3)

    # Check how argument:
    teams_kept = fuzzy_join(
        teams1,
        teams2,
        on=["basketball_teams", "teams_basketball"],
        analyzer=analyzer,
        match_type=match_type,
        match_threshold=0.1,
        how=how,
    )
    if how == "left":
        pd.testing.assert_frame_equal(teams_kept, teams1)
    if how == "right":
        assert teams_kept.shape == teams1.shape


@pytest.mark.parametrize(
    "analyzer, match_type, how, suffixes",
    [("a_blabla", "p_blabla", "k_blabla", ["a", "b", "c"]), (1, 26, 34, [1, 2, 3])],
)
def test_parameters_error(analyzer, match_type, how, suffixes):
    """Testing if correct errors are raised when wrong parameter values are given."""
    df1 = pd.DataFrame({"a": ["ana", "lala", "nana"], "b": [1, 2, 3]})
    df2 = pd.DataFrame({"a": ["anna", "lala", "ana", "sana"], "c": [5, 6, 7, 8]})
    with pytest.raises(
        ValueError,
        match=(
            f"analyzer should be either 'char', 'word' or 'char_wb', got {analyzer!r}"
        ),
    ):
        fuzzy_join(df1, df2, on=["a"], analyzer=analyzer)
    with pytest.raises(
        ValueError,
        match=f"match_type should be either 'nearest' or 'radius', got {match_type!r}",
    ):
        fuzzy_join(df1, df2, on=["a"], match_type=match_type)
    with pytest.raises(
        ValueError,
        match=f"how should be either 'left', 'right' or 'all', got {how!r}",
    ):
        fuzzy_join(df1, df2, on=["a"], how=how)
    with pytest.raises(
        ValueError, match="Invalid number of suffixes: expected 2, got 3"
    ):
        fuzzy_join(df1, df2, on="a", suffixes=suffixes)
    with pytest.raises(
        ValueError,
        match=(
            "value 'a' was specified for parameter 'on', which has invalid type,"
            " expected list of column names."
        ),
    ):
        fuzzy_join(df1, df2, on="a")
