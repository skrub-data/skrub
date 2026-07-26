import numpy as np
import pandas as pd
import pytest

from skrub._categorical_encoder import CategoricalEncoder


def test_pandas_frequent_and_rare_categories():
    X = pd.Series(
        ["paris"] * 4 + ["lyon"] * 3 + ["nice"] * 2 + ["rare_city"],
        name="city",
    )
    y = pd.Series([1, 0, 1, 1, 0, 1, 0, 1, 0, 1])

    encoder = CategoricalEncoder(min_frequency=3)
    transformed = encoder.fit_transform(X, y)

    assert "city__paris" in transformed.columns
    assert "city__lyon" in transformed.columns
    assert "city__rare_target" in transformed.columns

    assert "city__nice" not in transformed.columns
    assert "city__rare_city" not in transformed.columns


def test_one_hot_columns_are_binary():
    X = pd.Series(
        ["a"] * 4 + ["b"] * 3 + ["c"],
        name="feature",
    )
    y = pd.Series([0, 1, 0, 1, 1, 0, 1, 0])

    encoder = CategoricalEncoder(min_frequency=3)
    transformed = encoder.fit_transform(X, y)

    assert set(np.unique(transformed["feature__a"])) <= {0, 1}
    assert set(np.unique(transformed["feature__b"])) <= {0, 1}


def test_unseen_category_uses_global_target_mean():
    X_train = pd.Series(
        ["a"] * 4 + ["b"] * 2,
        name="feature",
    )
    y_train = pd.Series([0, 1, 0, 1, 1, 0])

    encoder = CategoricalEncoder(min_frequency=3)
    encoder.fit(X_train, y_train)

    X_test = pd.Series(["unknown"], name="feature")
    transformed = encoder.transform(X_test)

    expected = y_train.mean()

    assert transformed["feature__rare_target"].iloc[0] == pytest.approx(expected)


def test_transform_before_fit_raises():
    encoder = CategoricalEncoder(min_frequency=2)

    X = pd.Series(["a", "b"], name="feature")

    with pytest.raises(ValueError, match="not fitted"):
        encoder.transform(X)


def test_invalid_min_frequency_raises():
    X = pd.Series(["a", "b"], name="feature")
    y = pd.Series([0, 1])

    encoder = CategoricalEncoder(min_frequency=0)

    with pytest.raises(ValueError):
        encoder.fit(X, y)


def test_feature_names_out():
    X = pd.Series(
        ["a"] * 3 + ["b"],
        name="feature",
    )
    y = pd.Series([0, 1, 0, 1])

    encoder = CategoricalEncoder(min_frequency=2)
    encoder.fit(X, y)

    names = encoder.get_feature_names_out()

    assert "feature__a" in names
    assert "feature__rare_target" in names


def test_fit_returns_self():
    X = pd.Series(["a", "a", "b"], name="feature")
    y = pd.Series([0, 1, 0])

    encoder = CategoricalEncoder(min_frequency=2)

    result = encoder.fit(X, y)

    assert result is encoder


def test_output_keeps_pandas_index():
    X = pd.Series(
        ["a", "a", "b"],
        index=[10, 20, 30],
        name="feature",
    )
    y = pd.Series([0, 1, 0], index=[10, 20, 30])

    encoder = CategoricalEncoder(min_frequency=2)
    transformed = encoder.fit_transform(X, y)

    assert transformed.index.tolist() == [10, 20, 30]


def test_fit_transform_shape():
    X = pd.Series(
        ["a"] * 3 + ["b"] * 2 + ["c"],
        name="feature",
    )
    y = pd.Series([0, 1, 0, 1, 0, 1])

    encoder = CategoricalEncoder(min_frequency=3)
    transformed = encoder.fit_transform(X, y)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] == 2


def test_polars_support():
    pl = pytest.importorskip("polars")

    X = pl.Series(
        "feature",
        ["a", "a", "a", "b", "b", "c"],
    )
    y = pl.Series(
        "target",
        [0, 1, 0, 1, 0, 1],
    )

    encoder = CategoricalEncoder(min_frequency=3)
    transformed = encoder.fit_transform(X, y)

    assert isinstance(transformed, pl.DataFrame)

    assert "feature__a" in transformed.columns
    assert "feature__rare_target" in transformed.columns
