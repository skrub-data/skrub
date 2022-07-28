import pandas as pd
import pytest
from dirty_cat import FuzzyJoin

@pytest.mark.parametrize("analyzer, return_distance", [
    ('char', True),
    ('char_wb', True),
    ('word', True)
])
def test_fuzzy_join(analyzer, return_distance):
    teams1 = pd.DataFrame({'basketball_teams': ["LA Lakers", "Charlotte Hornets", "Polonia Warszawa", "Asseco", "Melbourne United (basketball)", "Partizan Belgrade","Liaoning FL", "P.A.O.K. BC", "New Orlean Plicans"]})
    teams2 = pd.DataFrame({'teams_basketball': ["Partizan BC", "New Orleans Pelicans", "Charlotte Hornets", "Polonia Warszawa (basketball)", "Real Madrid Baloncesto", "Los Angeles Lakers", "Asseco Gdynia", "PAOK BC", "Melbourne United", "Liaoning Flying Leopards"]})

    fj = FuzzyJoin(analyzer=analyzer)

    teams_joined, dist1 = fj.join(teams1, teams2, on=['basketball_teams', 'teams_basketball'], return_distance=return_distance)
    assert teams_joined.shape == (9, 2)
    assert dist1.shape == (9,1)

    teams_joined_2, dist2 = fj.join(teams2, teams1, on=['teams_basketball', 'basketball_teams'], return_distance=return_distance)
    # Joining is always done on the left table and thus takes it shape:
    assert teams_joined_2.shape == (10, 2)
    assert dist2.shape == (10,1)

    fj_2 = FuzzyJoin(analyzer=analyzer)
    teams_joined_3, dist3 = fj_2.join(teams2, teams1, on=['teams_basketball', 'basketball_teams'], return_distance=return_distance)

    pd.testing.assert_frame_equal(teams_joined_2, teams_joined_3)
    assert (dist3==dist2).all()
