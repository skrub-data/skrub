import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from skrub import InterpolationJoin


@pytest.fixture
def buildings():
    return pd.DataFrame(
        {"latitude": [1.0, 2.0], "longitude": [1.0, 2.0], "n_stories": [3, 7]}
    )


@pytest.fixture
def annual_avg_temp():
    return pd.DataFrame(
        {
            "latitude": [1.2, 0.9, 1.9, 1.7, 5.0, 5.0],
            "longitude": [0.8, 1.1, 1.8, 1.8, 5.0, 5.0],
            "avg_temp": [10.0, 11.0, 15.0, 16.0, 20.0, None],
            "climate": ["A", "A", "B", "B", "C", "C"],
        }
    )


@pytest.mark.parametrize("key", [["latitude", "longitude"], "latitude"])
@pytest.mark.parametrize("with_nulls", [False, True])
def test_interpolation_join(buildings, annual_avg_temp, key, with_nulls):
    if not with_nulls:
        annual_avg_temp = annual_avg_temp.fillna(0.0)
    transformed = InterpolationJoin(
        annual_avg_temp,
        key=key,
        regressor=KNeighborsRegressor(2),
        classifier=KNeighborsClassifier(2),
    ).fit_transform(buildings)
    assert_array_equal(transformed["avg_temp"].values, [10.5, 15.5])
    assert_array_equal(transformed["climate"].values, ["A", "B"])


def test_no_multioutput(buildings, annual_avg_temp):
    transformed = InterpolationJoin(
        annual_avg_temp,
        main_key=("latitude", "longitude"),
        aux_key=("latitude", "longitude"),
    ).fit_transform(buildings)
    assert transformed.shape == (2, 5)


def test_condition_choice():
    main = pd.DataFrame({"A": [0, 1, 2]})
    aux = pd.DataFrame({"A": [0, 1, 2], "rB": [2, 0, 1], "C": [10, 11, 12]})
    join = InterpolationJoin(
        aux, key="A", regressor=KNeighborsRegressor(1)
    ).fit_transform(main)
    assert_array_equal(join["C"].values, [10, 11, 12])

    join = InterpolationJoin(
        aux, main_key="A", aux_key="rB", regressor=KNeighborsRegressor(1)
    ).fit_transform(main)
    assert_array_equal(join["C"].values, [11, 12, 10])

    with pytest.raises(ValueError, match="Must pass EITHER"):
        join = InterpolationJoin(
            aux, main_key="A", regressor=KNeighborsRegressor(1)
        ).fit()

    with pytest.raises(ValueError, match="Can only pass"):
        join = InterpolationJoin(
            aux, key="A", main_key="A", regressor=KNeighborsRegressor(1)
        ).fit()

    with pytest.raises(ValueError, match="Can only pass"):
        join = InterpolationJoin(
            aux, key="A", main_key="A", aux_key="A", regressor=KNeighborsRegressor(1)
        ).fit()


def test_suffix():
    df = pd.DataFrame({"A": [0, 1], "B": [0, 1]})
    join = InterpolationJoin(
        df, key="A", suffix="_aux", regressor=KNeighborsRegressor(1)
    ).fit_transform(df)
    assert_array_equal(join.columns, ["A", "B", "B_aux"])


def test_mismatched_indexes():
    main = pd.DataFrame({"A": [0, 1]}, index=[1, 0])
    aux = pd.DataFrame({"A": [0, 1], "B": [10, 11]})
    join = InterpolationJoin(
        aux, key="A", regressor=KNeighborsRegressor(1)
    ).fit_transform(main)
    assert_array_equal(join["B"].values, [10, 11])
    assert_array_equal(join.index.values, [1, 0])


# expected to fail until we have a way to get the timestamp (only) from a date
# with the tablevectorizer
@pytest.mark.xfail
def test_join_on_date():
    sales = pd.DataFrame({"date": ["2023-09-20", "2023-09-29"], "n": [10, 15]})
    temp = pd.DataFrame(
        {"date": ["2023-09-09", "2023-10-01", "2024-09-21"], "temp": [-10, 10, 30]}
    )
    transformed = InterpolationJoin(
        temp,
        main_key="date",
        aux_key="date",
        regressor=KNeighborsRegressor(1),
    ).fit_transform(sales)
    assert_array_equal(transformed["temp"].values, [-10, 10])
