import re

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from skrub import GapEncoder
from skrub import _dataframe as sbd
from skrub._on_each_column import RejectColumn
from skrub.datasets import fetch_midwest_survey
from skrub.tests.utils import generate_data as _gen_data


@pytest.fixture
def generate_data(df_module):
    def generate(*args, as_list=True, **kwargs):
        del as_list
        data = _gen_data(*args, as_list=True, **kwargs)
        return df_module.make_column("some col", data)

    return generate


@pytest.mark.parametrize(
    ["hashing", "init", "rescale_W", "rescale_rho", "add_words"],
    [
        (False, "k-means++", True, False, True),
        (True, "random", False, True, False),
        (True, "k-means", True, True, False),
    ],
)
def test_analyzer(
    hashing,
    init,
    rescale_W,
    add_words,
    rescale_rho,
    generate_data,
):
    """
    Test if the output is different when the analyzer is 'word' or 'char'.
    If it is, no error ir raised.
    """
    n_samples = 70
    X = generate_data(n_samples, random_state=0)
    n_components = 10
    # Test first analyzer output:
    encoder = GapEncoder(
        n_components=n_components,
        hashing=hashing,
        init=init,
        analyzer="char",
        add_words=add_words,
        random_state=42,
        rescale_W=rescale_W,
        rescale_rho=rescale_rho,
    )
    encoder.fit(X)
    y1 = encoder.transform(X)
    s1 = encoder.score(X)

    # Test the other analyzer output:
    encoder = GapEncoder(
        n_components=n_components,
        hashing=hashing,
        init=init,
        analyzer="word",
        add_words=add_words,
        random_state=42,
        rescale_W=rescale_W,
        rescale_rho=rescale_rho,
    )
    encoder.fit(X)
    y2 = encoder.transform(X)
    s2 = encoder.score(X)

    # Test inequality between the word and char analyzers output:
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, y1, y2)
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, s1, s2)


@pytest.mark.parametrize(
    ["hashing", "init", "analyzer", "add_words", "verbose"],
    [
        (False, "k-means++", "word", True, False),
        (True, "random", "char", False, False),
        (True, "k-means", "char_wb", True, True),
    ],
)
def test_gap_encoder(
    hashing,
    init,
    analyzer,
    add_words,
    verbose,
    generate_data,
):
    n_samples = 70
    X = generate_data(n_samples, random_state=0)
    n_components = 10
    # Test output shape
    encoder = GapEncoder(
        n_components=n_components,
        hashing=hashing,
        init=init,
        analyzer=analyzer,
        add_words=add_words,
        verbose=verbose,
        random_state=42,
        rescale_W=True,
    )
    encoder.fit(X)
    y = encoder.transform(X)
    assert y.shape == (n_samples, n_components), str(y.shape)

    # Test L1-norm of topics W.
    l1_norm_W = np.abs(encoder.W_).sum(axis=1)
    np.testing.assert_array_almost_equal(l1_norm_W, np.ones(n_components))

    # Test same seed return the same output
    encoder = GapEncoder(
        n_components=n_components,
        hashing=hashing,
        init=init,
        analyzer=analyzer,
        add_words=add_words,
        random_state=42,
    )
    encoder.fit(X)
    y2 = encoder.transform(X)
    np.testing.assert_array_equal(y, y2)


@pytest.mark.parametrize(
    "add_words",
    [True, False],
)
def test_partial_fit(df_module, add_words, generate_data):
    n_samples = 70
    X = generate_data(n_samples, random_state=0)
    X2 = generate_data(n_samples - 10, random_state=1)
    X3 = generate_data(n_samples - 10, random_state=2)
    # Gap encoder with fit on one batch
    enc = GapEncoder(
        random_state=42, batch_size=n_samples, max_iter=1, add_words=add_words
    )
    X_enc = enc.fit_transform(X)
    # Gap encoder with partial fit
    enc = GapEncoder(random_state=42, add_words=add_words)
    enc.partial_fit(X)
    X_enc_partial = enc.transform(X)
    # Check if the encoded vectors are the same
    df_module.assert_frame_equal(X_enc, X_enc_partial)
    enc.partial_fit(X2)
    X_enc_partial2 = enc.transform(X3)
    with pytest.raises(AssertionError):
        df_module.assert_frame_equal(X_enc, X_enc_partial2)


def test_get_feature_names_out(generate_data):
    n_samples = 70
    X = generate_data(n_samples, random_state=0)
    enc = GapEncoder(random_state=42, n_components=3)
    enc.fit(X)
    feature_names = enc.get_feature_names_out()
    assert len(feature_names) == 3
    assert feature_names[0].startswith("some col: ")


def test_get_feature_names_out_no_words(df_module):
    # Test the GapEncoder get_feature_names_out when there are no words
    enc = GapEncoder(random_state=42)
    # A dataframe with words too short
    col = df_module.make_column("", 20 * ["a b c d"])

    enc.fit(col)
    # The difficulty here is that, in this specific case short words
    # should not be filtered out
    enc.get_feature_names_out()
    return


def test_get_feature_names_out_redundent(df_module):
    col = df_module.make_column("", 40 * ["aaa bbb cccc ddd"])
    enc = GapEncoder().fit(col)
    feat = enc.get_feature_names_out()
    assert re.match(r".* \(\d\)", feat[-1]) is not None
    assert len(set(feat)) == len(feat)


def test_overflow_error(df_module):
    np.seterr(over="raise", divide="raise")
    r = np.random.RandomState(0)
    X = r.randint(1e5, 1e6, size=8000).astype(str)
    X = df_module.make_column("", X)
    enc = GapEncoder(n_components=2, batch_size=1, max_iter=1, random_state=0)
    enc.fit(X)


def test_score(generate_data):
    n_samples = 70
    X = generate_data(n_samples, random_state=0)
    enc = GapEncoder(random_state=42)
    enc.fit(X)
    score_1 = enc.score(X)

    enc = GapEncoder(random_state=42)
    enc.fit(X)
    score_2 = enc.score(X)

    assert score_1 == score_2


def test_missing_values(df_module):
    """Test what happens when missing values are in the data."""
    if df_module.name == "pandas":
        m1, m2 = pd.NA, np.nan
    else:
        m1, m2 = None, None
    observations = ["alice", "bob", None, "alice", m1, m2]
    observations = df_module.make_column("", observations)
    enc = GapEncoder(n_components=3)
    enc.fit_transform(observations)
    enc.transform(observations)
    enc.partial_fit(observations)
    # non-regression for https://github.com/skrub-data/skrub/issues/921
    for null_idx in 2, 4, 5:
        assert sbd.is_null(observations)[null_idx]


def test_check_fitted_gap_encoder(df_module):
    """Test that calling transform before fit raises an error."""
    X = df_module.make_column("", ["alice", "bob"])
    enc = GapEncoder(n_components=2, random_state=42)
    with pytest.raises(NotFittedError):
        enc.transform(X)

    # Check that it works after fit
    enc.fit(X)
    enc.transform(X)


def test_small_sample(df_module):
    """Test that having n_samples < n_components raises an error."""
    X = df_module.make_column("", "alice bob".split())
    enc = GapEncoder(n_components=3, random_state=42)
    with pytest.raises(ValueError, match="should be >= n_components"):
        enc.fit_transform(X)


def test_transform_deterministic():
    """Non-regression test for #188."""
    dataset = fetch_midwest_survey()
    X_train, X_test = train_test_split(
        dataset.X["What_would_you_call_the_part_of_the_country_you_live_in_now"],
        random_state=0,
    )
    enc = GapEncoder(n_components=2, random_state=2)
    enc.fit_transform(X_train)
    topics1 = enc.get_feature_names_out()
    enc.transform(X_test)
    topics2 = enc.get_feature_names_out()
    assert_array_equal(topics1, topics2)


def test_max_no_improvements_none(generate_data):
    """Test that ``max_no_improvements=None`` works."""
    X = generate_data(300, random_state=0)
    enc_none = GapEncoder(n_components=2, max_no_improvement=None, random_state=42)
    enc_none.fit(X)


def test_bad_input_dtype(df_module):
    float_col = df_module.make_column("C", [2.2])
    with pytest.raises(RejectColumn, match="Column 'C' does not contain strings."):
        GapEncoder().fit(float_col)
    encoder = GapEncoder(n_components=2).fit(
        df_module.make_column("", "abc abc".split())
    )
    with pytest.raises(ValueError, match="Column 'C' does not contain strings.") as e:
        encoder.transform(float_col)
    assert e.type is ValueError


def test_output_pandas_index():
    s = pd.Series("one two two".split(), name="", index=[10, 20, 30])
    gap = GapEncoder(n_components=2).fit(s)
    s_test = pd.Series("one two two".split(), name="", index=[-11, 200, 32])
    out = gap.transform(s_test)
    assert out.index.tolist() == [-11, 200, 32]


def test_duplicate_topics(df_module):
    s = df_module.make_column("s", ["one", "two"] * 10)
    out = GapEncoder(n_components=3, random_state=0).fit_transform(s)
    assert sbd.column_names(out) == ["s: one, two", "s: two, one", "s: one, two (2)"]


def test_empty_column_name(df_module):
    s = df_module.make_column("", ["one", "two"] * 10)
    out = GapEncoder(n_components=3, random_state=0).fit_transform(s)
    assert sbd.column_names(out) == ["one, two", "two, one", "one, two (2)"]
