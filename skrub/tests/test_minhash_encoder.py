import random
from string import ascii_lowercase

import joblib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import skip_if_no_parallel

from skrub import MinHashEncoder
from skrub import _dataframe as sbd

from .utils import generate_data as _gen_data


@pytest.fixture
def generate_data(df_module):
    def generate(*args, as_list=True, **kwargs):
        del as_list
        data = _gen_data(*args, as_list=True, **kwargs)
        return df_module.make_column("some col", data)

    return generate


@pytest.mark.parametrize(
    ["hashing", "minmax_hash"],
    [
        ("fast", True),
        ("fast", False),
        ("murmur", False),
    ],
)
def test_minhash_encoder(df_module, hashing, minmax_hash):
    X = df_module.make_column("", ["al ice", "b ob", "bob and alice", "alice and bob"])
    # Test output shape
    encoder = MinHashEncoder(n_components=2, hashing=hashing)
    encoder.fit(X)
    y = encoder.transform(X)
    assert y.shape == (4, 2), str(y.shape)

    # Test that using the same seed returns the same output
    encoder2 = MinHashEncoder(n_components=2, hashing=hashing)
    encoder2.fit(X)
    y2 = encoder2.transform(X)
    assert_array_equal(y, y2)

    # Test min property
    if not minmax_hash:
        X_substring = [x[: x.find(" ")] for x in X]
        X_substring = df_module.make_column("", X_substring)
        encoder3 = MinHashEncoder(n_components=2, hashing=hashing)
        encoder3.fit(X_substring)
        y_substring = encoder3.transform(X_substring)
        np.testing.assert_array_less(y - y_substring, 0.001)


@pytest.mark.parametrize(
    ["hashing", "minmax_hash"],
    [
        ("fast", True),
        ("fast", False),
        ("murmur", False),
    ],
)
def test_encoder_params(generate_data, hashing, minmax_hash):
    X = generate_data(n_samples=20)
    enc = MinHashEncoder(
        n_components=50, hashing=hashing, minmax_hash=minmax_hash, ngram_range=(3, 3)
    )
    enc.fit(X)
    y = enc.transform(X)
    assert y.shape == (len(X), 50)


@pytest.mark.parametrize("hashing", ["fast", "murmur", "aaa"])
def test_missing_values(df_module, hashing):
    X = df_module.make_column(
        "",
        [
            "Red",
            pd.NA if df_module.description == "pandas-nullable-dtypes" else np.nan,
            "green",
            "blue",
            "green",
            "green",
            "blue",
            None,
        ],
    )
    encoder = MinHashEncoder(n_components=3, hashing=hashing, minmax_hash=False)

    if hashing == "aaa":
        with pytest.raises(ValueError, match=r"Got hashing="):
            encoder.fit_transform(X)
    else:
        y = encoder.fit_transform(X)
        assert y["_0"][1] == 0.0
        assert y["_0"][7] == 0.0
        # non-regression for https://github.com/skrub-data/skrub/issues/921
        assert sbd.is_null(X)[1]
        assert sbd.is_null(X)[7]


@pytest.mark.parametrize("hashing", ["fast", "murmur"])
def test_missing_values_none(df_module, hashing):
    # Test that "None" is also understood as a missing value
    a = df_module.make_column("", ["a", "  ", None, ""])

    enc = MinHashEncoder(hashing=hashing)
    d = enc.fit_transform(a)
    assert d["_0"][0] != 0.0
    assert d["_0"][1] != 0.0
    assert_array_equal(d["_0"][2], 0.0)
    assert_array_equal(d["_0"][3], 0.0)


def test_cache_overflow(df_module):
    # Regression test for cache overflow resulting in -1s in encoding
    def get_random_string(length):
        letters = ascii_lowercase
        result_str = "".join(random.choice(letters) for _ in range(length))
        return result_str

    encoder = MinHashEncoder(n_components=3)
    capacity = encoder._capacity
    raw_data = [get_random_string(10) for _ in range(capacity + 1)]
    raw_data = df_module.make_column("", raw_data)
    y = encoder.fit_transform(raw_data)
    assert (sbd.to_numpy(y["_0"]) != -1.0).all()


@skip_if_no_parallel
def test_parallelism(df_module):
    # Test that parallelism works
    encoder = MinHashEncoder(n_components=3, n_jobs=1)
    X = df_module.make_column("", ["a", "b", "c", "d", "e", "f", "g", "h"])
    y = encoder.fit_transform(X)
    for n_jobs in [None, 2, -1]:
        encoder = MinHashEncoder(n_components=3, n_jobs=n_jobs)
        y_parallel = encoder.fit_transform(X)
        df_module.assert_frame_equal(y, y_parallel)

    # Test with threading backend
    encoder = MinHashEncoder(n_components=3, n_jobs=2)
    with joblib.parallel_backend("threading"):
        y_threading = encoder.fit_transform(X)
    df_module.assert_frame_equal(y, y_threading)
    assert encoder.n_jobs == 2


DEFAULT_JOBLIB_BACKEND = joblib.parallel.get_active_backend()[0].__class__


class DummyBackend(DEFAULT_JOBLIB_BACKEND):  # type: ignore
    """
    A dummy backend used to check that specifying a backend works
    in MinHashEncoder.
    The `count` attribute is used to check that the backend is used.
    Copied from sklearn/ensemble/tests/test_forest.py
    """

    def __init__(self, *args, **kwargs):
        self.count = 0
        super().__init__(*args, **kwargs)

    def start_call(self):
        self.count += 1
        return super().start_call()


joblib.register_parallel_backend("testing", DummyBackend)


@skip_if_no_parallel
def test_backend_respected():
    """
    Test that the joblib backend is used.
    Copied from sklearn/ensemble/tests/test_forest.py
    """
    # Test that parallelism works
    encoder = MinHashEncoder(n_components=3, n_jobs=2)
    # this is not related to the dataframe so doesn't need to be tested for all
    # backends
    X = pd.Series(["a", "b", "c", "d", "e", "f", "g", "h"])

    with joblib.parallel_backend("testing") as (ba, n_jobs):
        encoder.fit_transform(X)

    assert ba.count > 0


def test_correct_arguments():
    # Test that the correct arguments are passed to the hashing function

    # this is not related to the dataframe so doesn't need to be tested for all
    # backends
    X = pd.Series(["a", "b", "c", "d", "e", "f", "g", "h"])
    # Write an incorrect value for the `hashing` argument
    with pytest.raises(ValueError, match=r"expected any of"):
        encoder = MinHashEncoder(n_components=3, hashing="incorrect")
        encoder.fit_transform(X)

    # Use minmax_hash with murmur hashing
    with pytest.raises(ValueError, match=r"minmax_hash encoding is not supported"):
        encoder = MinHashEncoder(n_components=2, minmax_hash=True, hashing="murmur")
        encoder.fit_transform(X)

    # Use minmax_hash with an odd number of components
    with pytest.raises(ValueError, match=r"n_components should be even"):
        encoder = MinHashEncoder(n_components=3, minmax_hash=True)
        encoder.fit_transform(X)


def test_check_fitted_minhash_encoder(df_module):
    """Test that calling transform before fit raises an error"""
    encoder = MinHashEncoder(n_components=3)
    X = df_module.make_column("some col", ["a", "b", "c", "d", "e", "f", "g", "h"])
    with pytest.raises(NotFittedError):
        encoder.transform(X)

    # Check that it works after fitting
    encoder.fit(X)
    encoder.transform(X)


def test_deterministic(df_module):
    """Test that the encoder is deterministic."""
    # TODO: add random state to encoder
    encoder1 = MinHashEncoder(n_components=4)
    encoder2 = MinHashEncoder(n_components=4)
    X = df_module.make_column("", ["a", "b", "c", "d", "e", "f", "g", "h"])
    encoded1 = encoder1.fit_transform(X)
    encoded2 = encoder2.fit_transform(X)
    df_module.assert_frame_equal(encoded1, encoded2)


def test_get_feature_names_out(df_module):
    """Test that ``get_feature_names_out`` returns the correct feature names."""
    encoder = MinHashEncoder(n_components=4)
    X = df_module.make_column(
        "col1",
        ["a", "b", "c", "d", "e", "f", "g", "h"],
    )
    encoder.fit(X)
    expected_columns = ["col1_0", "col1_1", "col1_2", "col1_3"]
    assert encoder.get_feature_names_out() == expected_columns
