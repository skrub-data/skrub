import re

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from skrub import InterpolationJoiner
from skrub import _dataframe as sbd


@pytest.fixture
def buildings(df_module):
    return df_module.make_dataframe(
        {"latitude": [1.0, 2.0], "longitude": [1.0, 2.0], "n_stories": [3, 7]}
    )


@pytest.fixture
def weather(df_module):
    return df_module.make_dataframe(
        {
            "latitude": [1.2, 0.9, 1.9, 1.7, 5.0, 5.0],
            "longitude": [0.8, 1.1, 1.8, 1.8, 5.0, 5.0],
            "avg_temp": [10.0, 11.0, 15.0, 16.0, 20.0, None],
            "climate": ["A", "A", "B", "B", "C", "C"],
        }
    )


def test_simple_join(buildings, weather):
    transformed = InterpolationJoiner(
        weather,
        main_key=("latitude", "longitude"),
        aux_key=("latitude", "longitude"),
    ).fit_transform(buildings)
    assert sbd.shape(transformed) == (2, 5)


def test_join_two_numeric_columns(buildings, weather):
    weather = sbd.with_columns(
        weather, **{"median_temp": [10.1, 10.9, 15.0, 16.2, 20.1, None]}
    )
    transformed = InterpolationJoiner(
        weather,
        main_key=("latitude", "longitude"),
        aux_key=("latitude", "longitude"),
    ).fit_transform(buildings)
    assert sbd.shape(transformed) == (2, 6)


def test_no_multioutput(buildings, weather):
    # Two str cols, not containing nulls, a model that handles them separately
    weather = sbd.with_columns(weather, **{"new_col": ["1", "1", "2", "2", "3", "3"]})
    transformed = InterpolationJoiner(
        weather,
        main_key=("latitude", "longitude"),
        aux_key=("latitude", "longitude"),
        classifier=LogisticRegression(),
    ).fit_transform(buildings)
    assert sbd.shape(transformed) == (2, 6)
    assert_array_equal(sbd.to_list(sbd.col(transformed, "climate")), ["A", "B"])
    assert_array_equal(sbd.to_list(sbd.col(transformed, "new_col")), ["1", "2"])


def test_multioutput(buildings, weather):
    # Two str cols, not containing nulls, a model that handles both at once
    weather = sbd.with_columns(weather, **{"new_col": ["1", "1", "2", "2", "3", "3"]})
    transformed = InterpolationJoiner(
        weather,
        main_key=("latitude", "longitude"),
        aux_key=("latitude", "longitude"),
        classifier=KNeighborsClassifier(2),
    ).fit_transform(buildings)
    assert sbd.shape(transformed) == (2, 6)
    assert_array_equal(sbd.to_list(sbd.col(transformed, "climate")), ["A", "B"])
    assert_array_equal(sbd.to_list(sbd.col(transformed, "new_col")), ["1", "2"])


@pytest.mark.parametrize("fill_nulls", [False, True])
def test_custom_predictors(buildings, weather, fill_nulls):
    if fill_nulls:
        weather = sbd.fill_nulls(weather, 0.0)
    transformed = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=KNeighborsClassifier(2),
    ).fit_transform(buildings)
    assert_array_equal(sbd.to_list(sbd.col(transformed, "avg_temp")), [10.5, 15.5])
    assert_array_equal(sbd.to_list(sbd.col(transformed, "climate")), ["A", "B"])


def test_custom_vectorizer(df_module):
    main = df_module.make_dataframe({"A": [0, 1]})
    aux = df_module.make_dataframe({"A": [11, 110], "B": [1, 0]})

    class Vectorizer(TransformerMixin, BaseEstimator):
        def fit(self, X):
            return self

        def transform(self, X):
            return X % 10

    transformed = InterpolationJoiner(
        aux,
        key="A",
        regressor=KNeighborsRegressor(1),
        vectorizer=Vectorizer(),
    ).fit_transform(main)
    assert_array_equal(sbd.col(transformed, "B"), [0, 1])


def test_duplicate_column(df_module):
    main = df_module.make_dataframe({"A": [0, 1, 2]})
    aux = df_module.make_dataframe({"A": [0, 1, 2], "C": [10, 11, 12]})
    transformed = InterpolationJoiner(
        aux, key="A", regressor=KNeighborsRegressor(1)
    ).fit_transform(main)
    assert_array_equal(sbd.to_list(sbd.col(transformed, "C")), [10, 11, 12])

    # Add column with the same name twice
    join_2 = InterpolationJoiner(
        aux, key="A", regressor=KNeighborsRegressor(1)
    ).fit_transform(transformed)
    assert sbd.shape(join_2) == (3, 3)
    assert re.match(r"C__skrub_[0-9a-f]+__", sbd.column_names(join_2)[2])


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
    transformed = InterpolationJoiner(
        df, key="A", suffix="_aux", regressor=KNeighborsRegressor(1)
    ).fit_transform(df)
    assert_array_equal(sbd.column_names(transformed), ["A", "B", "B_aux"])


def test_mismatched_indexes():
    main = pd.DataFrame({"A": [0, 1]}, index=[20, 30])
    aux = pd.DataFrame({"A": [0, 1], "B": [10, 11]}, index=[0, 1])
    transformed = InterpolationJoiner(
        aux, key="A", regressor=KNeighborsRegressor(1)
    ).fit_transform(main)
    assert_array_equal(sbd.to_list(sbd.col(transformed, "B")), [10, 11])
    # Index of main is kept
    assert_array_equal(transformed.index.values, [20, 30])


def test_fit_on_none(df_module):
    # X is hardly used in fit so it should be ok to fit without a main table
    aux = df_module.make_dataframe({"A": [0, 1], "B": [10, 11]})
    joiner = InterpolationJoiner(aux, key="A", regressor=KNeighborsRegressor(1)).fit(
        None
    )
    main = df_module.make_dataframe({"A": [0, 1]})
    transformed = joiner.transform(main)
    assert_array_equal(sbd.to_list(sbd.col(transformed, "B")), [10, 11])


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
    assert_array_equal(sbd.to_list(sbd.col(transformed, "temp")), [-10, 10])


class FailFit(DummyClassifier):
    def fit(self, X, y):
        raise ValueError("FailFit failed")


def test_fit_failures(buildings, weather):
    weather = sbd.with_columns(weather, **{"climate": ["A"] * sbd.shape(weather)[0]})
    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailFit(),
        on_estimator_failure="pass",
    )
    transformed = joiner.fit_transform(buildings)
    assert_array_equal(sbd.to_list(sbd.col(transformed, "avg_temp")), [10.5, 15.5])
    # Only numerical columns have been added
    assert transformed.shape == (2, 4)
    assert sbd.column_names(transformed) == [
        "latitude",
        "longitude",
        "n_stories",
        "avg_temp",
    ]

    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailFit(),
        on_estimator_failure="warn",
    )
    with pytest.warns(UserWarning, match="(?s)Estimators failed.*climate"):
        transformed = joiner.fit_transform(buildings)
    assert_array_equal(sbd.to_list(sbd.col(transformed, "avg_temp")), [10.5, 15.5])
    assert sbd.shape(transformed) == (2, 4)
    assert sbd.column_names(transformed) == [
        "latitude",
        "longitude",
        "n_stories",
        "avg_temp",
    ]

    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailFit(),
        on_estimator_failure="raise",
    )
    with pytest.raises(ValueError, match="FailFit failed"):
        transformed = joiner.fit_transform(buildings)


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
    transformed = joiner.fit_transform(buildings)
    assert_array_equal(sbd.to_list(sbd.col(transformed, "avg_temp")), [10.5, 15.5])
    assert sbd.is_null(sbd.col(transformed, "climate")).all()
    assert sbd.dtype(sbd.col(transformed, "climate")) == sbd.dtype(
        sbd.col(weather, "climate")
    )
    assert sbd.shape(transformed) == (2, 5)

    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailPredict(),
        on_estimator_failure="warn",
    )
    with pytest.warns(UserWarning, match="(?s)Prediction failed.*climate"):
        transformed = joiner.fit_transform(buildings)
    assert_array_equal(sbd.to_list(sbd.col(transformed, "avg_temp")), [10.5, 15.5])
    assert sbd.is_null(sbd.col(transformed, "climate")).all()
    assert sbd.dtype(sbd.col(transformed, "climate")) == sbd.dtype(
        sbd.col(weather, "climate")
    )
    assert sbd.shape(transformed) == (2, 5)

    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=KNeighborsRegressor(2),
        classifier=FailPredict(),
        on_estimator_failure="raise",
    )
    with pytest.raises(Exception, match="FailPredict failed"):
        transformed = joiner.fit_transform(buildings)


def test_transform_failures_dtype(buildings, weather):
    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=FailPredict(),
        classifier=DummyClassifier(),
        on_estimator_failure="pass",
    )
    transformed = joiner.fit_transform(buildings)
    assert sbd.is_null(sbd.col(transformed, "avg_temp")).all()
    assert sbd.dtype(sbd.col(transformed, "avg_temp")) == sbd.dtype(
        sbd.col(weather, "avg_temp")
    )
    assert sbd.shape(transformed) == (2, 5)

    joiner = InterpolationJoiner(
        weather,
        key=["latitude", "longitude"],
        regressor=DummyRegressor(),
        classifier=FailPredict(),
        on_estimator_failure="pass",
    )
    transformed = joiner.fit_transform(buildings)
    assert sbd.is_null(sbd.col(transformed, "climate")).all()
    assert sbd.dtype(sbd.col(transformed, "climate")) == sbd.dtype(
        sbd.col(weather, "climate")
    )
    assert sbd.shape(transformed) == (2, 5)
