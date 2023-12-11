import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from skrub import GapEncoder, TableVectorizer
from skrub._dataframe._polars import POLARS_SETUP
from skrub._dataframe._test_utils import is_module_polars
from skrub.datasets import fetch_midwest_survey
from skrub.tests.utils import generate_data

MODULES = [pd]

if POLARS_SETUP:
    import polars as pl

    MODULES.append(pl)


@pytest.mark.parametrize(
    ["hashing", "init", "rescale_W", "rescale_rho", "add_words"],
    [
        (False, "k-means++", True, False, True),
        (True, "random", False, True, False),
        (True, "k-means", True, True, False),
    ],
)
def test_analyzer(
    hashing: bool,
    init: str,
    rescale_W: bool,
    add_words: bool,
    rescale_rho: bool,
    n_samples: int = 70,
):
    """
    Test if the output is different when the analyzer is 'word' or 'char'.
    If it is, no error ir raised.
    """
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
    hashing: bool,
    init: str,
    analyzer: str,
    add_words: bool,
    verbose: bool,
    n_samples: int = 70,
):
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
    assert y.shape == (n_samples, n_components * X.shape[1]), str(y.shape)

    # Test L1-norm of topics W.
    for col_enc in encoder.fitted_models_:
        l1_norm_W = np.abs(col_enc.W_).sum(axis=1)
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


@pytest.mark.parametrize("px", MODULES)
def test_input_type(px):
    # Numpy array with one column
    X = np.array([["alice"], ["bob"]])
    enc = GapEncoder(n_components=2, random_state=42)
    X_enc_array = enc.fit_transform(X)
    # List
    X2 = [["alice"], ["bob"]]
    enc = GapEncoder(n_components=2, random_state=42)
    X_enc_list = enc.fit_transform(X2)
    # Check if the encoded vectors are the same
    np.testing.assert_array_equal(X_enc_array, X_enc_list)

    # Numpy array with two columns
    X = np.array([["alice", "charlie"], ["bob", "delta"]])
    enc = GapEncoder(n_components=2, random_state=42)
    X_enc_array = enc.fit_transform(X)
    # Dataframe with two columns
    df = px.DataFrame(X)
    enc = GapEncoder(n_components=2, random_state=42)
    X_enc_df = enc.fit_transform(df)
    # Check if the encoded vectors are the same
    np.testing.assert_array_equal(X_enc_array, X_enc_df)


@pytest.mark.parametrize("px", MODULES)
@pytest.mark.parametrize(
    "add_words",
    [True, False],
)
def test_partial_fit(px, add_words: bool, n_samples: int = 70):
    X = generate_data(n_samples, random_state=0)
    X2 = px.DataFrame(generate_data(n_samples - 10, random_state=1))
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
    np.testing.assert_almost_equal(X_enc, X_enc_partial)
    enc.partial_fit(X2)
    X_enc_partial2 = enc.transform(X3)
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, X_enc, X_enc_partial2
    )


def test_get_feature_names_out(n_samples=70):
    X = generate_data(n_samples, random_state=0)
    enc = GapEncoder(random_state=42)
    enc.fit(X)
    feature_names_1 = enc.get_feature_names_out()
    feature_names_2 = enc.get_feature_names_out()
    for topic_labels in [feature_names_1, feature_names_2]:
        # Check number of labels
        assert len(topic_labels) == enc.n_components * X.shape[1]
        # Test different parameters for col_names
        topic_labels_2 = enc.get_feature_names_out(col_names="auto")
        assert topic_labels_2[0] == "col0: " + topic_labels[0]
        topic_labels_3 = enc.get_feature_names_out(col_names=["abc", "def"])
        assert topic_labels_3[0] == "abc: " + topic_labels[0]
    return


def test_get_feature_names_out_no_words():
    # Test the GapEncoder get_feature_names_out when there are no words
    enc = GapEncoder(random_state=42)
    # A dataframe with words too short
    df = pd.DataFrame(
        20
        * [
            [
                "a b c d",
            ],
        ],
    )

    enc.fit(df)
    # The difficulty here is that, in this specific case short words
    # should not be filtered out
    enc.get_feature_names_out()
    return


def test_get_feature_names_out_redundent():
    # With the following dataframe, the GapEncoder can produce feature names
    # that have the same name, which leads duplicated features names,
    # which themselves lead to errors in the TableVectorizer
    # get_feature_names_out() method.
    df = pd.DataFrame(
        40
        * [
            [
                "aaa bbb cccc ddd",
            ],
        ],
    )

    tv = TableVectorizer(cardinality_threshold=1)
    tv.fit(df)
    tv.get_feature_names_out()


def test_overflow_error():
    np.seterr(over="raise", divide="raise")
    r = np.random.RandomState(0)
    X = r.randint(1e5, 1e6, size=(8000, 1)).astype(str)
    enc = GapEncoder(n_components=2, batch_size=1, max_iter=1, random_state=0)
    enc.fit(X)


def test_score(n_samples: int = 70):
    X1 = generate_data(n_samples, random_state=0)
    X2 = np.hstack([X1, X1])
    enc = GapEncoder(random_state=42)
    enc.fit(X1)
    score_X1 = enc.score(X1)
    enc.fit(X2)
    score_X2 = enc.score(X2)
    # Check that two identical columns give the same score
    assert score_X1 * 2 == score_X2


@pytest.mark.parametrize("px", MODULES)
@pytest.mark.parametrize(
    "missing",
    ["zero_impute", "error", "aaa"],
)
def test_missing_values(px, missing: str):
    """Test what happens when missing values are in the data"""
    if is_module_polars(px):
        pytest.xfail(
            reason=(
                "'TypeError: '<' not supported between instances of 'DataTypeClass' and"
                " 'str'' raised because of pl.Null"
            )
        )
    observations = [
        ["alice", "bob"],
        [pd.NA, "alice"],
        ["bob", None],
        ["alice", "charlie"],
        [np.nan, "alice"],
    ]
    observations = np.array(observations, dtype=object)
    enc = GapEncoder(handle_missing=missing, n_components=3)
    if missing == "error":
        with pytest.raises(ValueError, match="Input data contains missing values"):
            enc.fit_transform(observations)
    elif missing == "zero_impute":
        enc.fit_transform(observations)
        enc.partial_fit(observations)
    else:
        with pytest.raises(
            ValueError,
            match=r"handle_missing should be either "
            r"'error' or 'zero_impute', got 'aaa'",
        ):
            enc.fit_transform(observations)


def test_check_fitted_gap_encoder():
    """Test that calling transform before fit raises an error"""
    X = np.array([["alice"], ["bob"]])
    enc = GapEncoder(n_components=2, random_state=42)
    with pytest.raises(NotFittedError):
        enc.transform(X)

    # Check that it works after fit
    enc.fit(X)
    enc.transform(X)


def test_small_sample():
    """Test that having n_samples < n_components raises an error"""
    X = np.array([["alice"], ["bob"]])
    enc = GapEncoder(n_components=3, random_state=42)
    with pytest.raises(ValueError, match="should be >= n_components"):
        enc.fit_transform(X)


def test_transform_deterministic():
    """Non-regression test for #188"""
    dataset = fetch_midwest_survey()
    X_train, X_test = train_test_split(
        dataset.X[["What_would_you_call_the_part_of_the_country_you_live_in_now"]],
        random_state=0,
    )
    enc = GapEncoder(n_components=2, random_state=2)
    enc.fit_transform(X_train)
    topics1 = enc.get_feature_names_out()
    enc.transform(X_test)
    topics2 = enc.get_feature_names_out()
    assert_array_equal(topics1, topics2)


def test_max_no_improvements_none():
    """Test that max_no_improvements=None works"""
    X = generate_data(300, random_state=0)
    enc_none = GapEncoder(n_components=2, max_no_improvement=None, random_state=42)
    enc_none.fit(X)
