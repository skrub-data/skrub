import pandas as pd
import pytest
from dirty_cat import fuzzy_join


@pytest.mark.parametrize(
    "analyzer, precision", [("char", "nearest"), ("char_wb", "radius")]
)
def test_fuzzy_join(analyzer, precision, return_distance=True):
    """ Testing if fuzzy_join gives joining results as expected. """
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
                "New Orleans Pelicans"
            ]
        }
    )

    teams_joined, dist1 = fuzzy_join(
        teams1,
        teams2,
        on=["basketball_teams", "teams_basketball"],
        return_distance=return_distance,
        analyzer=analyzer, precision=precision,
        precision_threshold=0.1
    )
    # Check correct shapes of outputs
    assert teams_joined.shape == (9, 2)
    assert dist1.shape == (9, 1)
    assert (teams_joined == ground_truth).all()[1]
    teams_joined_2, dist2 = fuzzy_join(
        teams2,
        teams1,
        on=["teams_basketball", "basketball_teams"],
        return_distance=return_distance,
        analyzer=analyzer, precision=precision,
        precision_threshold=0.1
    )
    # Joining is always done on the left table and thus takes it shape:
    assert teams_joined_2.shape == (10, 2)
    assert dist2.shape == (10, 1)

    teams_joined_3, dist3 = fuzzy_join(
        teams2,
        teams1,
        on=["teams_basketball", "basketball_teams"],
        return_distance=return_distance,
        analyzer=analyzer, precision=precision,
        precision_threshold=0.1
    )
    # Check invariability of joining:
    pd.testing.assert_frame_equal(teams_joined_2, teams_joined_3)
    assert (dist3 == dist2).all()

@pytest.mark.parametrize(
    "analyzer, precision, keep", [("a_blabla", "p_blabla", "k_blabla"), (1, 26, 34)]
)
def test_parameters_error(analyzer, precision, keep):
    """ Testing if correct errors are raised when wrong parameter values are given. """
    df1 = pd.DataFrame({'a': ['ana', 'lala', 'nana'], 'b': [1, 2, 3]})
    df2 = pd.DataFrame({'a': ['anna', 'lala', 'ana', 'sana'], 'c': [5, 6, 7, 8]})
    with pytest.raises(ValueError, match=f"analyzer should be either 'char', 'word' or 'char_wb', got {analyzer}"):
        fuzzy_join(df1, df2, on=['a'], analyzer=analyzer)
    with pytest.raises(ValueError, match=f"precision should be either 'nearest' or 'radius', got {precision}"):
        fuzzy_join(df1, df2, on=['a'], precision=precision)
    with pytest.raises(ValueError, match=f"keep should be either 'left', 'right' or 'all', got {keep}"):
        fuzzy_join(df1, df2, on=['a'], keep=keep)
    with pytest.raises(ValueError, match=f"keep should be either 'left', 'right' or 'all', got {keep}"):
        fuzzy_join(df1, df2, on=['a'], keep=keep)
    with pytest.raises(ValueError, match="value a was specified for parameter 'on', which has invalid type, expected list of column names."):
        fuzzy_join(df1, df2, on='a')
