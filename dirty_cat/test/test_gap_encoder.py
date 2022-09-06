import numpy as np
import pandas as pd
import pytest
from sklearn import __version__ as sklearn_version
from sklearn.datasets import fetch_20newsgroups

from dirty_cat import GapEncoder
from dirty_cat.utils import Version


def test_analyzer():
    """
    Test if the output is different when the analyzer is 'word' or 'char'.
    If it is, no error ir raised.
    """
    add_words = False
    n_samples = 70
    X_txt = fetch_20newsgroups(subset="train")["data"][:n_samples]
    X = np.array([X_txt, X_txt]).T
    n_components = 10
    # Test first analyzer output:
    encoder = GapEncoder(
        n_components=n_components,
        init="k-means++",
        analyzer="char",
        add_words=add_words,
        random_state=42,
        rescale_W=True,
    )
    encoder.fit(X)
    y = encoder.transform(X)

    # Test the other analyzer output:
    encoder = GapEncoder(
        n_components=n_components,
        init="k-means++",
        analyzer="word",
        add_words=add_words,
        random_state=42,
    )
    encoder.fit(X)
    y2 = encoder.transform(X)

    # Test inequality between the word and char analyzers output:
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, y, y2)


@pytest.mark.parametrize(
    "hashing, init, analyzer, add_words",
    [
        (False, "k-means++", "word", True),
        (True, "random", "char", False),
        (True, "k-means", "char_wb", True),
    ],
)
def test_gap_encoder(
    hashing: bool, init: str, analyzer: str, add_words: bool, n_samples: int = 70
) -> None:
    X_txt = fetch_20newsgroups(subset="train")["data"][:n_samples]
    X = np.array([X_txt, X_txt]).T
    n_components = 10
    # Test output shape
    encoder = GapEncoder(
        n_components=n_components,
        hashing=hashing,
        init=init,
        analyzer=analyzer,
        add_words=add_words,
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


def test_input_type() -> None:
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
    # Pandas dataframe with two columns
    df = pd.DataFrame(X)
    enc = GapEncoder(n_components=2, random_state=42)
    X_enc_df = enc.fit_transform(df)
    # Check if the encoded vectors are the same
    np.testing.assert_array_equal(X_enc_array, X_enc_df)


def test_partial_fit(n_samples=70) -> None:
    X_txt = fetch_20newsgroups(subset="train")["data"][:n_samples]
    X = np.array([X_txt, X_txt]).T
    # Gap encoder with fit on one batch
    enc = GapEncoder(random_state=42, batch_size=n_samples, max_iter=1)
    X_enc = enc.fit_transform(X)
    # Gap encoder with partial fit
    enc = GapEncoder(random_state=42)
    enc.partial_fit(X)
    X_enc_partial = enc.transform(X)
    # Check if the encoded vectors are the same
    np.testing.assert_almost_equal(X_enc, X_enc_partial)


def test_get_feature_names_out(n_samples=70) -> None:
    X_txt = fetch_20newsgroups(subset="train")["data"][:n_samples]
    X = np.array([X_txt, X_txt]).T
    enc = GapEncoder(random_state=42)
    enc.fit(X)
    # Expect a warning if sklearn >= 1.0
    if Version(sklearn_version) < "1.0":
        feature_names_1 = enc.get_feature_names()
    else:
        with pytest.warns(DeprecationWarning):
            feature_names_1 = enc.get_feature_names()
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


def test_overflow_error() -> None:
    np.seterr(over="raise", divide="raise")
    r = np.random.RandomState(0)
    X = r.randint(1e5, 1e6, size=(8000, 1)).astype(str)
    enc = GapEncoder(
        n_components=2, batch_size=1, min_iter=1, max_iter=1, random_state=0
    )
    enc.fit(X)


def test_score(n_samples: int = 70) -> None:
    X_txt = fetch_20newsgroups(subset="train")["data"][:n_samples]
    X1 = np.array(X_txt)[:, None]
    X2 = np.hstack([X1, X1])
    enc = GapEncoder(random_state=42)
    enc.fit(X1)
    score_X1 = enc.score(X1)
    enc.fit(X2)
    score_X2 = enc.score(X2)
    # Check that two identical columns give the same score
    assert score_X1 * 2 == score_X2


@pytest.mark.parametrize("missing", ["zero_impute", "error", "aaa"])
def test_missing_values(missing: str) -> None:
    observations = [
        ["alice", "bob"],
        ["bob", "alice"],
        ["bob", np.nan],
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


if __name__ == "__main__":
    print("test_analyzer")
    test_analyzer()
    print("test_analyzer passed")

    print("start test_gap_encoder")
    test_gap_encoder(hashing=True, init="k-means++", analyzer="char", add_words=False)
    print("passed")

    print("start test_input_type")
    test_input_type()
    print("passed")

    print("start test_partial_fit")
    test_partial_fit()
    print("passed")

    print("start test_get_feature_names_out")
    test_get_feature_names_out()
    print("passed")

    print("start test_overflow_error")
    test_overflow_error()
    print("passed")

    print("start test_score")
    test_score()
    print("test_score passed")

    print("Done")
