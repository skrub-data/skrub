import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from skrub import InterpolationJoiner
from skrub._dataframe import _common as ns


@pytest.fixture
def buildings():
    return pd.DataFrame(
        {"latitude": [1.0, 2.0], "longitude": [1.0, 2.0], "n_stories": [3, 7]}
    )


@pytest.fixture
def weather():
    return pd.DataFrame(
        {
            "latitude": [1.2, 0.9, 1.9, 1.7, 5.0, 5.0],
            "longitude": [0.8, 1.1, 1.8, 1.8, 5.0, 5.0],
            "avg_temp": [10.0, 11.0, 15.0, 16.0, 20.0, None],
            "climate": ["A", "A", "B", "B", "C", "C"],
        }
    )


def test_simple_join(df_module, buildings, weather):
    weather = df_module.DataFrame(weather)
    buildings = df_module.DataFrame(buildings)
    transformed = InterpolationJoiner(
        weather,
        main_key=("latitude", "longitude"),
        aux_key=("latitude", "longitude"),
    ).fit_transform(buildings)
    assert ns.shape(transformed) == (2, 5)


@pytest.mark.parametrize("fill_nulls", [False, True])
def test_custom_predictors(df_module, buildings, weather, fill_nulls):
    if fill_nulls:
        weather = ns.fill_nulls(weather, 0.0)
    weather = df_module.DataFrame(weather)
    buildings = df_module.DataFrame(buildings)
    transformed = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=KNeighborsClassifier(2),
    ).fit_transform(buildings)
    assert_array_equal(ns.to_list(ns.col(transformed, "avg_temp")), [10.5, 15.5])
    assert_array_equal(ns.to_list(ns.col(transformed, "climate")), ["A", "B"])


def test_custom_vectorizer(df_module):
    main = df_module.make_dataframe({"A": [0, 1]})
    aux = df_module.make_dataframe({"A": [11, 110], "B": [1, 0]})

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
    assert_array_equal(ns.col(join, "B"), [0, 1])


def test_condition_choice(df_module):
    main = df_module.make_dataframe({"A": [0, 1, 2]})
    aux = df_module.make_dataframe({"A": [0, 1, 2], "rB": [2, 0, 1], "C": [10, 11, 12]})
    join = InterpolationJoiner(
        aux, key="A", regressor=KNeighborsRegressor(1)
    ).fit_transform(main)
    assert_array_equal(ns.to_list(ns.col(join, "C")), [10, 11, 12])

    # Add column with the same name twice
    join = InterpolationJoiner(
        aux, main_key="A", aux_key="rB", regressor=KNeighborsRegressor(1)
    ).fit_transform(main)


def test_wrong_key(df_module):
    main = df_module.make_dataframe({"A": [0, 1, 2]})
    aux = df_module.make_dataframe({"A": [0, 1, 2], "C": [10, 11, 12]})

    with pytest.raises(ValueError, match="Must pass either"):
        InterpolationJoiner(aux, main_key="A", regressor=KNeighborsRegressor(1)).fit(
            main
        )

    with pytest.raises(ValueError, match="Can only pass"):
        InterpolationJoiner(
            aux, key="A", main_key="A", regressor=KNeighborsRegressor(1)
        ).fit(main)

    with pytest.raises(ValueError, match="Can only pass"):
        InterpolationJoiner(
            aux, key="A", main_key="A", aux_key="A", regressor=KNeighborsRegressor(1)
        ).fit(main)


def test_suffix(df_module):
    df = df_module.make_dataframe({"A": [0, 1], "B": [0, 1]})
    join = InterpolationJoiner(
        df, key="A", suffix="_aux", regressor=KNeighborsRegressor(1)
    ).fit_transform(df)
    assert_array_equal(ns.column_names(join), ["A", "B", "B_aux"])


def test_fit_on_none(df_module):
    # X is hardly used in fit so it should be ok to fit without a main table
    aux = df_module.make_dataframe({"A": [0, 1], "B": [10, 11]})
    joiner = InterpolationJoiner(aux, key="A", regressor=KNeighborsRegressor(1)).fit(
        None
    )
    main = df_module.make_dataframe({"A": [1, 0]})
    join = joiner.transform(main)
    assert_array_equal(ns.to_list(ns.col(join, "B")), [11, 10])


def test_join_on_date(df_module):
    sales = df_module.make_dataframe(
        {"date": ["2023-09-20", "2023-09-29"], "n": [10, 15]}
    )
    temp = df_module.make_dataframe(
        {"date": ["2023-09-09", "2023-10-01", "2024-09-21"], "temp": [-10, 10, 30]}
    )
    transformed = (
        InterpolationJoiner(
            temp,
            main_key="date",
            aux_key="date",
            regressor=KNeighborsRegressor(1),
        )
        .set_params(vectorizer__datetime__resolution=None)
        .fit_transform(sales)
    )
    assert_array_equal(ns.to_list(ns.col(transformed, "temp")), [-10, 10])


class FailFit(DummyClassifier):
    def fit(self, X, y):
        raise ValueError("FailFit failed")


def test_fit_failures(df_module, buildings, weather):
    weather = df_module.DataFrame(weather)
    buildings = df_module.DataFrame(buildings)
    weather = ns.with_columns(weather, **{"climate": ["A"] * ns.shape(weather)[0]})
    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailFit(),
        on_estimator_failure="pass",
    )
    join = joiner.fit_transform(buildings)
    assert_array_equal(ns.to_list(ns.col(join, "avg_temp")), [10.5, 15.5])
    # Only numerical columns have been added
    assert join.shape == (2, 4)
    assert ns.column_names(join) == ["latitude", "longitude", "n_stories", "avg_temp"]

    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailFit(),
        on_estimator_failure="warn",
    )
    with pytest.warns(UserWarning, match="(?s)Estimators failed.*climate"):
        join = joiner.fit_transform(buildings)
    assert_array_equal(ns.to_list(ns.col(join, "avg_temp")), [10.5, 15.5])
    assert ns.shape(join) == (2, 4)
    assert ns.column_names(join) == ["latitude", "longitude", "n_stories", "avg_temp"]

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


def test_transform_failures(df_module, buildings, weather):
    weather = df_module.DataFrame(weather)
    buildings = df_module.DataFrame(buildings)
    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailPredict(),
        on_estimator_failure="pass",
    )
    join = joiner.fit_transform(buildings)
    assert_array_equal(ns.to_list(ns.col(join, "avg_temp")), [10.5, 15.5])
    assert ns.is_null(ns.col(join, "climate")).all()
    # TODO: fix object dtype for polars
    assert ns.dtype(ns.col(join, "climate")) == object
    assert ns.shape(join) == (2, 5)

    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailPredict(),
        on_estimator_failure="warn",
    )
    with pytest.warns(UserWarning, match="(?s)Prediction failed.*climate"):
        join = joiner.fit_transform(buildings)
    assert_array_equal(ns.to_list(ns.col(join, "avg_temp")), [10.5, 15.5])
    assert ns.is_null(ns.col(join, "climate")).all()
    # TODO: fix object dtype for polars
    assert ns.dtype(ns.col(join, "climate")) == object
    assert ns.shape(join) == (2, 5)

    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailPredict(),
        on_estimator_failure="raise",
    )
    with pytest.raises(Exception, match="FailPredict failed"):
        join = joiner.fit_transform(buildings)


def test_transform_failures_dtype(df_module, buildings, weather):
    weather = df_module.DataFrame(weather)
    buildings = df_module.DataFrame(buildings)
    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=FailPredict(),
        classifier=DummyClassifier(),
        on_estimator_failure="pass",
    )
    join = joiner.fit_transform(buildings)
    assert ns.is_null(ns.col(join, "avg_temp")).all()
    assert ns.dtype(ns.col(join, "avg_temp")) == "float64"
    assert ns.shape(join) == (2, 5)

    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=DummyRegressor(),
        classifier=FailPredict(),
        on_estimator_failure="pass",
    )
    join = joiner.fit_transform(buildings)
    assert ns.is_null(ns.col(join, "climate")).all()
    assert ns.dtype(ns.col(join, "climate")) == object
    assert ns.shape(join) == (2, 5)
