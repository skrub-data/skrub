import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsRegressor

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
        }
    )


@pytest.mark.parametrize("join_on", [["latitude", "longitude"], "latitude"])
@pytest.mark.parametrize("with_nulls", [False, True])
@pytest.mark.parametrize("with_vectorizer", [False, True])
def test_interpolation_join(
    buildings, annual_avg_temp, join_on, with_nulls, with_vectorizer
):
    if not with_nulls:
        annual_avg_temp = annual_avg_temp.fillna(0.0)
    params = {} if with_vectorizer else {"vectorizer": None}
    transformed = InterpolationJoin(
        annual_avg_temp,
        left_on=join_on,
        right_on=join_on,
        regressor=KNeighborsRegressor(2),
        **params,
    ).fit_transform(buildings)
    assert (transformed["avg_temp"].values == [10.5, 15.5]).all()


def test_no_multioutput(buildings, annual_avg_temp):
    transformed = InterpolationJoin(
        annual_avg_temp,
        left_on=("latitude", "longitude"),
        right_on=("latitude", "longitude"),
    ).fit_transform(buildings)
    assert transformed.shape == (2, 4)


def test_condition_choice():
    left = pd.DataFrame({"A": [0, 1, 2]})
    right = pd.DataFrame({"A": [0, 1, 2], "rB": [2, 0, 1], "C": [10, 11, 12]})
    join = InterpolationJoin(
        right, on="A", regressor=KNeighborsRegressor(1)
    ).fit_transform(left)
    assert (join["C"].values == [10, 11, 12]).all()

    join = InterpolationJoin(
        right, left_on="A", right_on="rB", regressor=KNeighborsRegressor(1)
    ).fit_transform(left)
    assert (join["C"].values == [11, 12, 10]).all()

    with pytest.raises(ValueError):
        join = InterpolationJoin(
            right, left_on="A", regressor=KNeighborsRegressor(1)
        ).fit()

    with pytest.raises(ValueError):
        join = InterpolationJoin(
            right, on="A", left_on="A", regressor=KNeighborsRegressor(1)
        ).fit()

    with pytest.raises(ValueError):
        join = InterpolationJoin(
            right, on="A", left_on="A", right_on="A", regressor=KNeighborsRegressor(1)
        ).fit()


def test_suffix():
    df = pd.DataFrame({"A": [0, 1], "B": [0, 1]})
    join = InterpolationJoin(
        df, on="A", suffix="_right", regressor=KNeighborsRegressor(1)
    ).fit_transform(df)
    assert (join.columns == ["A", "B", "B_right"]).all()


def test_mismatched_indexes():
    left = pd.DataFrame({"A": [0, 1]}, index=[1, 0])
    right = pd.DataFrame({"A": [0, 1], "B": [10, 11]})
    join = InterpolationJoin(
        right, on="A", regressor=KNeighborsRegressor(1)
    ).fit_transform(left)
    assert (join["B"].values == [10, 11]).all()
    assert (join.index.values == [1, 0]).all()


# expected to fail until we have a way to get the timestamp (only) from a date
# with the tablevectorizer
@pytest.mark.xfail
@pytest.mark.parametrize("with_vectorizer", [False, True])
def test_join_on_date(with_vectorizer):
    sales = pd.DataFrame({"date": ["2023-09-20", "2023-09-29"], "n": [10, 15]})
    temp = pd.DataFrame(
        {"date": ["2023-09-09", "2023-10-01", "2024-09-21"], "temp": [-10, 10, 30]}
    )
    params = {} if with_vectorizer else {"vectorizer": None}
    transformed = InterpolationJoin(
        temp,
        left_on="date",
        right_on="date",
        regressor=KNeighborsRegressor(1),
        **params,
    ).fit_transform(sales)
    assert (transformed["temp"].values == [-10, 10]).all()
