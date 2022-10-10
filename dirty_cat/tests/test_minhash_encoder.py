import random
from string import ascii_lowercase

import numpy as np
import pandas as pd
import pytest

from dirty_cat import MinHashEncoder


@pytest.mark.parametrize(
    "hashing, minmax_hash", [("fast", True), ("fast", False), ("murmur", False)]
)
def test_MinHashEncoder(hashing, minmax_hash) -> None:
    X = np.array(["al ice", "b ob", "bob and alice", "alice and bob"])[:, None]
    # Test output shape
    encoder = MinHashEncoder(n_components=2, hashing=hashing)
    encoder.fit(X)
    y = encoder.transform(X)
    assert y.shape == (4, 2), str(y.shape)
    assert len(set(y[0])) == 2

    # Test same seed return the same output
    encoder2 = MinHashEncoder(2, hashing=hashing)
    encoder2.fit(X)
    y2 = encoder2.transform(X)
    np.testing.assert_array_equal(y, y2)

    # Test min property
    if not minmax_hash:
        X_substring = [x[: x.find(" ")] for x in X[:, 0]]
        X_substring = np.array(X_substring)[:, None]
        encoder3 = MinHashEncoder(2, hashing=hashing)
        encoder3.fit(X_substring)
        y_substring = encoder3.transform(X_substring)
        np.testing.assert_array_less(y - y_substring, 0.001)


def test_multiple_columns() -> None:
    """
    This test aims at verifying that fitting multiple columns
    with the MinHashEncoder will not produce an error, and will
    encode the column independently.
    """
    X = pd.DataFrame(
        [
            ("bird", "parrot"),
            ("bird", "nightingale"),
            ("mammal", "monkey"),
            ("mammal", np.nan),
        ],
        columns=("class", "type"),
    )
    X1 = X[["class"]]
    X2 = X[["type"]]
    fit1 = MinHashEncoder(n_components=30).fit_transform(X1)
    fit2 = MinHashEncoder(n_components=30).fit_transform(X2)
    fit = MinHashEncoder(n_components=30).fit_transform(X)
    assert np.array_equal(
        np.array([fit[:, :30], fit[:, 30:60]]), np.array([fit1, fit2])
    )


def test_input_type() -> None:
    # Numpy array
    X = np.array(["alice", "bob"])[:, None]
    enc = MinHashEncoder(n_components=2)
    enc.fit_transform(X)
    # List
    X = [["alice"], ["bob"]]
    enc = MinHashEncoder(n_components=2)
    enc.fit_transform(X)


@pytest.mark.parametrize(
    "hashing, minmax_hash", [("fast", True), ("fast", False), ("murmur", False)]
)
def test_encoder_params(hashing, minmax_hash) -> None:
    MAX_LIMIT = 255  # extended ASCII Character set
    i = 0
    str_list = []
    for i in range(100):
        random_string = "aa"
        for _ in range(100):
            random_integer = random.randint(0, MAX_LIMIT)
            random_string += chr(random_integer)
            if random_integer < 50:
                random_string += "  "
        i += 1
        str_list += [random_string]
    X = np.array(str_list).reshape(100, 1)
    enc = MinHashEncoder(
        n_components=50, hashing=hashing, minmax_hash=minmax_hash, ngram_range=(3, 3)
    )
    enc.fit(X)
    y = enc.transform(X)
    assert y.shape == (len(X), 50)
    X2 = np.array([["a", "", "c"]]).T
    y2 = enc.transform(X2)
    assert y2.shape == (len(X2), 50)


@pytest.mark.parametrize(
    "input_type, missing, hashing",
    [
        ["numpy", "error", "fast"],
        ["pandas", "zero_impute", "murmur"],
        ["numpy", "zero_impute", "fast"],
        ["numpy", "zero_impute", "aaa"],
        ["numpy", "aaaa", "fast"],
    ],
)
def test_missing_values(input_type: str, missing: str, hashing: str) -> None:
    X = ["Red", np.nan, "green", "blue", "green", "green", "blue", float("nan")]
    n = 3
    z = np.zeros(n)

    if input_type == "numpy":
        X = np.array(X, dtype=object)[:, None]
    elif input_type == "pandas":
        X = pd.DataFrame(X)

    encoder = MinHashEncoder(
        n_components=n, hashing=hashing, minmax_hash=False, handle_missing=missing
    )

    if hashing == "aaa":
        with pytest.raises(ValueError, match=r"Got hashing="):
            encoder.fit_transform(X)
    else:
        if missing == "error":
            if input_type in ["numpy", "pandas"]:
                with pytest.raises(
                    ValueError, match=r"Found missing values in input data; set"
                ):
                    encoder.fit_transform(X)
        elif missing == "zero_impute":
            y = encoder.fit_transform(X)
            assert np.array_equal(y[1], z)
            assert np.array_equal(y[-1], z)
        else:
            with pytest.raises(ValueError, match=r"Got handle_missing="):
                encoder.fit_transform(X)
    return


def test_missing_values_none():
    # Test that "None" is also understood as a missing value
    a = np.array([["a", "b", None, "c"]], dtype=object).T

    enc = MinHashEncoder()
    d = enc.fit_transform(a)
    np.testing.assert_array_equal(d[2], 0)


def test_cache_overflow() -> None:
    # Regression test for cache overflow resulting in -1s in encoding
    def get_random_string(length):
        letters = ascii_lowercase
        result_str = "".join(random.choice(letters) for _ in range(length))
        return result_str

    encoder = MinHashEncoder(n_components=3)
    capacity = encoder._capacity
    raw_data = [get_random_string(10) for _ in range(capacity + 1)]
    raw_data = np.array(raw_data)[:, None]
    y = encoder.fit_transform(raw_data)

    assert len(y[y == -1.0]) == 0
