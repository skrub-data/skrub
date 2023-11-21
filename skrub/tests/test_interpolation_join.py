import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from skrub import InterpolationJoiner
from skrub._dataframe._polars import POLARS_SETUP
from skrub._dataframe._test_utils import is_module_polars

MODULES = [pd]

if POLARS_SETUP:
    import polars as pl

    MODULES.append(pl)


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


@pytest.mark.parametrize("px", MODULES)
@pytest.mark.parametrize("key", [["latitude", "longitude"], "latitude"])
@pytest.mark.parametrize("with_nulls", [False, True])
def test_interpolation_join(px, buildings, weather, key, with_nulls):
    if is_module_polars(px):
        pytest.xfail(
            reason=(
                "In polars, DataFrame.drop() got an unexpected keyword argument 'axis'"
            )
        )
    weather = weather.fillna(0.0)
    weather = px.DataFrame(weather)
    buildings = px.DataFrame(buildings)
    transformed = InterpolationJoiner(
        weather,
        key=key,
        regressor=KNeighborsRegressor(2),
        classifier=KNeighborsClassifier(2),
    ).fit_transform(buildings)
    assert_array_equal(transformed["avg_temp"].values, [10.5, 15.5])
    assert_array_equal(transformed["climate"].values, ["A", "B"])


@pytest.mark.parametrize("px", MODULES)
def test_vectorizer(px):
    if is_module_polars(px):
        pytest.xfail(
            reason=(
                "In polars, DataFrame.drop() got an unexpected keyword argument 'axis'"
            )
        )
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


@pytest.mark.parametrize("px", MODULES)
def test_no_multioutput(px, buildings, weather):
    if is_module_polars(px):
        pytest.xfail(
            reason=(
                "In polars, DataFrame.drop() got an unexpected keyword argument 'axis'"
            )
        )
    weather = px.DataFrame(weather)
    buildings = px.DataFrame(buildings)
    transformed = InterpolationJoiner(
        weather,
        main_key=("latitude", "longitude"),
        aux_key=("latitude", "longitude"),
    ).fit_transform(buildings)
    assert transformed.shape == (2, 5)


@pytest.mark.parametrize("px", MODULES)
def test_condition_choice(px):
    if is_module_polars(px):
        pytest.xfail(
            reason=(
                "In polars, DataFrame.drop() got an unexpected keyword argument 'axis'"
            )
        )
    main = pd.DataFrame({"A": [0, 1, 2]})
    main = px.DataFrame(main)
    aux = pd.DataFrame({"A": [0, 1, 2], "rB": [2, 0, 1], "C": [10, 11, 12]})
    aux = px.DataFrame(aux)
    join = InterpolationJoiner(
        aux, key="A", regressor=KNeighborsRegressor(1)
    ).fit_transform(main)
    assert_array_equal(join["C"].values, [10, 11, 12])

    join = InterpolationJoiner(
        aux, main_key="A", aux_key="rB", regressor=KNeighborsRegressor(1)
    ).fit_transform(main)
    assert_array_equal(join["C"].values, [11, 12, 10])

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


@pytest.mark.parametrize("px", MODULES)
def test_suffix(px):
    if is_module_polars(px):
        pytest.xfail(
            reason=(
                "In polars, DataFrame.drop() got an unexpected keyword argument 'axis'"
            )
        )
    df = pd.DataFrame({"A": [0, 1], "B": [0, 1]})
    df = px.DataFrame(df)
    join = InterpolationJoiner(
        df, key="A", suffix="_aux", regressor=KNeighborsRegressor(1)
    ).fit_transform(df)
    assert_array_equal(join.columns, ["A", "B", "B_aux"])


@pytest.mark.parametrize("px", MODULES)
def test_mismatched_indexes(px):
    if is_module_polars(px):
        pytest.xfail(
            reason=(
                "In polars, DataFrame.drop() got an unexpected keyword argument 'axis'"
            )
        )
    main = pd.DataFrame({"A": [0, 1]}, index=[1, 0])
    main = px.DataFrame(main)
    aux = pd.DataFrame({"A": [0, 1], "B": [10, 11]})
    aux = px.DataFrame(aux)
    join = InterpolationJoiner(
        aux, key="A", regressor=KNeighborsRegressor(1)
    ).fit_transform(main)
    assert_array_equal(join["B"].values, [10, 11])
    assert_array_equal(join.index.values, [1, 0])


@pytest.mark.parametrize("px", MODULES)
def test_fit_on_none(px):
    # X is hardly used in fit so it should be ok to fit without a main table
    if is_module_polars(px):
        pytest.xfail(
            reason=(
                "In polars, DataFrame.drop() got an unexpected keyword argument 'axis'"
            )
        )
    aux = pd.DataFrame({"A": [0, 1], "B": [10, 11]})
    aux = px.DataFrame(aux)
    joiner = InterpolationJoiner(aux, key="A", regressor=KNeighborsRegressor(1)).fit(
        None
    )
    main = pd.DataFrame({"A": [0, 1]}, index=[1, 0])
    main = px.DataFrame(main)
    join = joiner.transform(main)
    assert_array_equal(join["B"].values, [10, 11])
    assert_array_equal(join.index.values, [1, 0])


@pytest.mark.parametrize("px", MODULES)
def test_join_on_date(px):
    if is_module_polars(px):
        pytest.xfail(
            reason=(
                "In polars, DataFrame.drop() got an unexpected keyword argument 'axis'"
            )
        )
    sales = pd.DataFrame({"date": ["2023-09-20", "2023-09-29"], "n": [10, 15]})
    sales = px.DataFrame(sales)
    temp = pd.DataFrame(
        {"date": ["2023-09-09", "2023-10-01", "2024-09-21"], "temp": [-10, 10, 30]}
    )
    temp = px.DataFrame(temp)
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
    assert_array_equal(transformed["temp"].values, [-10, 10])


class FailFit(DummyClassifier):
    def fit(self, X, y):
        raise ValueError("FailFit failed")


@pytest.mark.parametrize("px", MODULES)
def test_fit_failures(px, buildings, weather):
    if is_module_polars(px):
        pytest.xfail(
            reason=(
                "In polars, DataFrame.drop() got an unexpected keyword argument 'axis'"
            )
        )
    weather = px.DataFrame(weather)
    buildings = px.DataFrame(buildings)
    weather["climate"] = "A"
    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailFit(),
        on_estimator_failure="pass",
    )
    join = joiner.fit_transform(buildings)
    assert_array_equal(join["avg_temp"].values, [10.5, 15.5])
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
    assert_array_equal(join["avg_temp"].values, [10.5, 15.5])
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


@pytest.mark.parametrize("px", MODULES)
def test_transform_failures(px, buildings, weather):
    if is_module_polars(px):
        pytest.xfail(
            reason=(
                "In polars, DataFrame.drop() got an unexpected keyword argument 'axis'"
            )
        )
    weather = px.DataFrame(weather)
    buildings = px.DataFrame(buildings)
    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailPredict(),
        on_estimator_failure="pass",
    )
    join = joiner.fit_transform(buildings)
    assert_array_equal(join["avg_temp"].values, [10.5, 15.5])
    assert join["climate"].isnull().all()
    assert join["climate"].dtype == object
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
    assert_array_equal(join["avg_temp"].values, [10.5, 15.5])
    assert join["climate"].isnull().all()
    assert join["climate"].dtype == object
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


@pytest.mark.parametrize("px", MODULES)
def test_transform_failures_dtype(px, buildings, weather):
    if is_module_polars(px):
        pytest.xfail(
            reason=(
                "In polars, DataFrame.drop() got an unexpected keyword argument 'axis'"
            )
        )
    weather = px.DataFrame(weather)
    buildings = px.DataFrame(buildings)
    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=FailPredict(),
        classifier=DummyClassifier(),
        on_estimator_failure="pass",
    )
    join = joiner.fit_transform(buildings)
    assert join["avg_temp"].isnull().all()
    assert join["avg_temp"].dtype == "float64"
    assert join.shape == (2, 5)

    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=DummyRegressor(),
        classifier=FailPredict(),
        on_estimator_failure="pass",
    )
    join = joiner.fit_transform(buildings)
    assert join["climate"].isnull().all()
    assert join["climate"].dtype == object
    assert join.shape == (2, 5)
