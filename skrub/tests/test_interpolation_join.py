import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from skrub import InterpolationJoiner


@pytest.fixture
def buildings(px):
    return px.DataFrame(
        {"latitude": [1.0, 2.0], "longitude": [1.0, 2.0], "n_stories": [3, 7]}
    )


@pytest.fixture
def weather(px):
    return px.DataFrame(
        {
            "latitude": [1.2, 0.9, 1.9, 1.7, 5.0, 5.0],
            "longitude": [0.8, 1.1, 1.8, 1.8, 5.0, 5.0],
            "avg_temp": [10.0, 11.0, 15.0, 16.0, 20.0, None],
            "climate": ["A", "A", "B", "B", "C", "C"],
        }
    )


@pytest.mark.parametrize("key", [["latitude", "longitude"], "latitude"])
@pytest.mark.parametrize("with_nulls", [False, True])
def test_interpolation_join(buildings, weather, key, with_nulls):
    if not with_nulls:
        weather = weather.__dataframe_consortium_standard__().fill_null(0.0).dataframe
    transformed = InterpolationJoiner(
        weather,
        key=key,
        regressor=KNeighborsRegressor(2),
        classifier=KNeighborsClassifier(2),
        suffix="_w",
    ).fit_transform(buildings)
    assert_array_equal(transformed["avg_temp_w"].to_numpy(), [10.5, 15.5])
    assert_array_equal(transformed["climate_w"].to_numpy(), ["A", "B"])


def test_vectorizer(px):
    main = px.DataFrame({"A": [0, 1]})
    aux = px.DataFrame({"A": [11, 110], "B": [1, 0]})

    class Vectorizer(TransformerMixin, BaseEstimator):
        def fit(self, X):
            return self

        def transform(self, X):
            return X % 10

    join = InterpolationJoiner(
        aux,
        key="A",
        regressor=KNeighborsRegressor(1),
        vectorizer=Vectorizer(),
    ).fit_transform(main)
    assert_array_equal(join["B"], [0, 1])


def test_no_multioutput(buildings, weather):
    transformed = InterpolationJoiner(
        weather,
        main_key=("latitude", "longitude"),
        aux_key=("latitude", "longitude"),
    ).fit_transform(buildings)
    assert transformed.shape == (2, 5)


def test_condition_choice(px):
    main = px.DataFrame({"A": [0, 1, 2]})
    aux = px.DataFrame({"A": [0, 1, 2], "rB": [2, 0, 1], "C": [10, 11, 12]})
    join = InterpolationJoiner(
        aux, key="A", regressor=KNeighborsRegressor(1)
    ).fit_transform(main)
    assert_array_equal(join["C"].to_numpy(), [10, 11, 12])

    join = InterpolationJoiner(
        aux, main_key="A", aux_key="rB", regressor=KNeighborsRegressor(1), suffix="_r"
    ).fit_transform(main)
    assert_array_equal(join["C_r"].to_numpy(), [11, 12, 10])

    with pytest.raises(ValueError, match="Must pass EITHER"):
        join = InterpolationJoiner(
            aux, main_key="A", regressor=KNeighborsRegressor(1)
        ).fit(None)

    with pytest.raises(ValueError, match="Can only pass"):
        join = InterpolationJoiner(
            aux, key="A", main_key="A", regressor=KNeighborsRegressor(1)
        ).fit(None)

    with pytest.raises(ValueError, match="Can only pass"):
        join = InterpolationJoiner(
            aux, key="A", main_key="A", aux_key="A", regressor=KNeighborsRegressor(1)
        ).fit(None)


def test_suffix(px):
    df = px.DataFrame({"A": [0, 1], "B": [0, 1]})
    join = InterpolationJoiner(
        df, key="A", suffix="_aux", regressor=KNeighborsRegressor(1)
    ).fit_transform(df)
    assert_array_equal(join.columns, ["A", "B", "B_aux"])


def test_mismatched_indexes():
    main = pd.DataFrame({"A": [0, 1]}, index=[1, 0])
    aux = pd.DataFrame({"A": [0, 1], "B": [10, 11]})
    join = InterpolationJoiner(
        aux, key="A", regressor=KNeighborsRegressor(1)
    ).fit_transform(main)
    assert_array_equal(join["B"].to_numpy(), [10, 11])
    assert_array_equal(join.index.to_numpy(), [1, 0])


def test_fit_on_none(px):
    # X is hardly used in fit so it should be ok to fit without a main table
    aux = px.DataFrame({"A": [0, 1], "B": [10, 11]})
    joiner = InterpolationJoiner(aux, key="A", regressor=KNeighborsRegressor(1)).fit(
        None
    )
    main = px.DataFrame({"A": [0, 1]})
    if isinstance(main, pd.DataFrame):
        main.index = [1, 0]
    join = joiner.transform(main)
    assert_array_equal(join["B"].to_numpy(), [10, 11])
    if isinstance(main, pd.DataFrame):
        assert_array_equal(join.index.to_numpy(), [1, 0])


def test_join_on_date(px):
    sales = px.DataFrame({"date": ["2023-09-20", "2023-09-29"], "n": [10, 15]})
    temp = px.DataFrame(
        {"date": ["2023-09-09", "2023-10-01", "2024-09-21"], "temp": [-10, 10, 30]}
    )
    transformed = (
        InterpolationJoiner(
            temp,
            main_key="date",
            aux_key="date",
            regressor=KNeighborsRegressor(1),
        )
        .set_params(vectorizer__datetime_transformer__resolution=None)
        .fit_transform(sales)
    )
    assert_array_equal(transformed["temp"].to_numpy(), [-10, 10])


class FailFit(DummyClassifier):
    def fit(self, X, y):
        raise ValueError("FailFit failed")


def test_fit_failures(buildings, weather):
    try:
        weather["climate"] = "A"
    except TypeError:
        import polars as pl

        weather = weather.with_columns(pl.lit("A").alias("climate"))
    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailFit(),
        on_estimator_failure="pass",
    )
    join = joiner.fit_transform(buildings)
    assert_array_equal(join["avg_temp"].to_numpy(), [10.5, 15.5])
    assert join.shape == (2, 4)

    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailFit(),
        on_estimator_failure="warn",
    )
    with pytest.warns(UserWarning, match="(?s)Estimators failed.*climate"):
        join = joiner.fit_transform(buildings)
    assert_array_equal(join["avg_temp"].to_numpy(), [10.5, 15.5])
    assert join.shape == (2, 4)

    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailFit(),
        on_estimator_failure="raise",
    )
    with pytest.raises(ValueError, match="FailFit failed"):
        join = joiner.fit_transform(buildings)


class FailPredict(DummyClassifier):
    def predict(self, X):
        raise ValueError("FailPredict failed")


def test_transform_failures(buildings, weather):
    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailPredict(),
        on_estimator_failure="pass",
    )
    join = joiner.fit_transform(buildings)
    assert_array_equal(join["avg_temp"].to_numpy(), [10.5, 15.5])
    assert join["climate"].__column_consortium_standard__().is_null().all()
    assert join["climate"].dtype == weather["climate"].dtype
    assert join.shape == (2, 5)

    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailPredict(),
        on_estimator_failure="warn",
    )
    with pytest.warns(UserWarning, match="(?s)Prediction failed.*climate"):
        join = joiner.fit_transform(buildings)
    assert_array_equal(join["avg_temp"].to_numpy(), [10.5, 15.5])
    assert join["climate"].__column_consortium_standard__().is_null().all()
    assert join["climate"].dtype == weather["climate"].dtype
    assert join.shape == (2, 5)

    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailPredict(),
        on_estimator_failure="raise",
    )
    with pytest.raises(Exception, match="FailPredict failed"):
        join = joiner.fit_transform(buildings)


def test_transform_failures_dtype(buildings, weather):
    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=FailPredict(),
        classifier=DummyClassifier(),
        on_estimator_failure="pass",
    )
    join = joiner.fit_transform(buildings)
    assert join["avg_temp"].__column_consortium_standard__().is_null().all()
    assert join["avg_temp"].dtype == join["avg_temp"].dtype
    assert join.shape == (2, 5)

    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=DummyRegressor(),
        classifier=FailPredict(),
        on_estimator_failure="pass",
    )
    join = joiner.fit_transform(buildings)
    assert join["climate"].__column_consortium_standard__().is_null().all()
    assert join["climate"].dtype == weather["climate"].dtype
    assert join.shape == (2, 5)
