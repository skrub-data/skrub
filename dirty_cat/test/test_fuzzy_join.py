import pandas as pd
import pytest
from dirty_cat import FuzzyJoin


@pytest.mark.parametrize(
    "analyzer, precision", [("char", "closest"), ("char_wb", "2dball")]
)
def test_fuzzy_join(analyzer, precision, return_distance=True):
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

    fj = FuzzyJoin(analyzer=analyzer, precision=precision,
                   precision_threshold=0.5)

    teams_joined, dist1 = fj.join(
        teams1,
        teams2,
        on=["basketball_teams", "teams_basketball"],
        return_distance=return_distance,
    )

    # Check correct shapes of outputs
    assert teams_joined.shape == (9, 2)
    assert dist1.shape == (9, 1)

    # Check performance of FuzzyJoin
    assert (teams_joined == ground_truth).all()[1]

    teams_joined_2, dist2 = fj.join(
        teams2,
        teams1,
        on=["teams_basketball", "basketball_teams"],
        return_distance=return_distance,
    )

    # Joining is always done on the left table and thus takes it shape:
    assert teams_joined_2.shape == (10, 2)
    assert dist2.shape == (10, 1)

    fj_2 = FuzzyJoin(analyzer=analyzer)
    teams_joined_3, dist3 = fj_2.join(
        teams2,
        teams1,
        on=["teams_basketball", "basketball_teams"],
        return_distance=return_distance,
    )

    # Check invariability of joining:
    pd.testing.assert_frame_equal(teams_joined_2, teams_joined_3)
    assert (dist3 == dist2).all()
