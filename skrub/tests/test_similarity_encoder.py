from typing import Callable, Optional

import numpy as np
import numpy.testing
import pytest
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import NotFittedError

from skrub import SimilarityEncoder
from skrub._similarity_encoder import get_kmeans_prototypes, ngram_similarity_matrix
from skrub._string_distances import ngram_similarity
from skrub._utils import parse_version


def test_specifying_categories() -> None:
    # When creating a new SimilarityEncoder:
    # - if categories = 'auto', the categories are the sorted, unique training
    # set observations (for each column)
    # - if categories is a list (of lists), the categories for each column are
    # each item in the list

    # In this test, we first find the sorted, unique categories in the training
    # set, and create a SimilarityEncoder by giving it explicitly the computed
    # categories. The test consists in making sure the transformed observations
    # given by this encoder are equal to the transformed observations in the
    # case of a SimilarityEncoder created with categories = 'auto'

    observations = [["bar"], ["foo"]]
    categories = [["bar", "foo"]]

    sim_enc_with_cat = SimilarityEncoder(categories=categories, ngram_range=(2, 3))
    sim_enc_auto_cat = SimilarityEncoder(ngram_range=(2, 3))

    feature_matrix_with_cat = sim_enc_with_cat.fit_transform(observations)
    feature_matrix_auto_cat = sim_enc_auto_cat.fit_transform(observations)

    assert np.allclose(feature_matrix_auto_cat, feature_matrix_with_cat)


def test_fast_ngram_similarity() -> None:
    vocabulary = [["bar", "foo"]]
    observations = [["foo"], ["baz"]]

    sim_enc = SimilarityEncoder(ngram_range=(2, 2), categories=vocabulary)

    sim_enc.fit(observations)
    feature_matrix = sim_enc.transform(observations, fast=False)
    feature_matrix_fast = sim_enc.transform(observations, fast=True)

    assert np.allclose(feature_matrix, feature_matrix_fast)


def test_parameters() -> None:
    X = [["foo"], ["baz"]]
    X2 = [["foo"], ["bar"]]
    with pytest.raises(ValueError, match=r"Got handle_unknown="):
        SimilarityEncoder(handle_unknown="bb").fit(X)
    with pytest.raises(ValueError, match=r"Got hashing_dim="):
        SimilarityEncoder(hashing_dim="bb").fit(X)
    with pytest.raises(ValueError, match=r"Got categories="):
        SimilarityEncoder(categories="bb")
    with pytest.raises(ValueError, match=r"Unsorted categories "):
        SimilarityEncoder(categories=[["cat2", "cat1"], ["cat3", "cat4"]]).fit(X)
    with pytest.raises(ValueError, match=r"Found unknown categories "):
        SimilarityEncoder(categories=[["fooo", "loo"]], handle_unknown="error").fit(X)
    with pytest.raises(ValueError, match=r"Found unknown categories "):
        sim = SimilarityEncoder(categories=[["baz", "foo"]], handle_unknown="error")
        sim.fit(X)
        sim.transform(X2)


def _test_missing_values(input_type: str, missing: str) -> None:
    observations = [["a", "b"], ["b", "a"], ["b", np.nan], ["a", "c"], [np.nan, "a"]]
    encoded = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )

    if input_type == "numpy":
        observations = np.array(observations, dtype=object)
    elif input_type == "pandas":
        pd = pytest.importorskip("pandas")
        observations = pd.DataFrame(observations)

    sim_enc = SimilarityEncoder(handle_missing=missing)
    if missing == "error":
        with pytest.raises(ValueError, match=r"Found missing values in input"):
            sim_enc.fit_transform(observations)
    elif missing == "":
        ans = sim_enc.fit_transform(observations)
        assert np.allclose(encoded, ans)
    else:
        with pytest.raises(ValueError, match=r"expected any of"):
            sim_enc.fit_transform(observations)
        return


def _test_missing_values_transform(input_type: str, missing: str) -> None:
    observations = [["a", "b"], ["b", "a"], ["b", "b"], ["a", "c"], ["c", "a"]]
    test_observations = [
        ["a", "b"],
        ["b", "a"],
        ["b", np.nan],
        ["a", "c"],
        [np.nan, "a"],
    ]
    encoded = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )

    if input_type == "numpy":
        test_observations = np.array(test_observations, dtype=object)
    elif input_type == "pandas":
        pd = pytest.importorskip("pandas")
        test_observations = pd.DataFrame(test_observations)

    sim_enc = SimilarityEncoder(handle_missing=missing)
    if missing == "error":
        sim_enc.fit_transform(observations)
        with pytest.raises(ValueError, match=r"Found missing values in input"):
            sim_enc.transform(test_observations)
    elif missing == "":
        sim_enc.fit_transform(observations)
        ans = sim_enc.transform(test_observations)
        assert np.allclose(encoded, ans)


def _test_similarity(
    similarity_f: Callable,
    hashing_dim: Optional[int] = None,
    categories: str = "auto",
    n_prototypes: int = None,
) -> None:
    if n_prototypes is None:
        X = np.array(["aa", "aaa", "aaab"]).reshape(-1, 1)
        X_test = np.array([["Aa", "aAa", "aaa", "aaab", " aaa  c"]]).reshape(-1, 1)

        model = SimilarityEncoder(
            hashing_dim=hashing_dim,
            categories=categories,
            n_prototypes=n_prototypes,
            ngram_range=(3, 3),
        )

        X_test_enc = model.fit(X).transform(X_test)

        ans = np.zeros((len(X_test), len(X)))
        for i, x_t in enumerate(X_test.reshape(-1)):
            for j, x in enumerate(X.reshape(-1)):
                ans[i, j] = similarity_f(x_t, x, 3)
        numpy.testing.assert_almost_equal(X_test_enc, ans)
    else:
        X = np.array(
            ["aac", "aaa", "aaab", "aaa", "aaab", "aaa", "aaab", "aaa"]
        ).reshape(-1, 1)
        X_test = np.array([["Aa", "aAa", "aaa", "aaab", " aaa  c"]]).reshape(-1, 1)

        if categories == "auto":
            with pytest.warns(UserWarning, match=r"n_prototypes parameter ignored"):
                SimilarityEncoder(
                    categories=categories,
                    n_prototypes=n_prototypes,
                )
        try:
            model = SimilarityEncoder(
                hashing_dim=hashing_dim,
                categories=categories,
                n_prototypes=n_prototypes,
                random_state=42,
                ngram_range=(3, 3),
            )
        except ValueError as e:
            assert (
                e.__str__()
                == "n_prototypes expected None or a positive non null integer. "
            )
            return

        X_test_enc = model.fit(X).transform(X_test)
        if n_prototypes == 1:
            assert model.categories_ == ["aaa"]
        elif n_prototypes == 2:
            a = [np.array(["aaa", "aaab"], dtype="<U4")]
            assert np.array_equal(a, model.categories_)
        elif n_prototypes == 3:
            a = [np.array(["aaa", "aaab", "aac"], dtype="<U4")]
            assert np.array_equal(a, model.categories_)

        ans = np.zeros((len(X_test), len(np.array(model.categories_).reshape(-1))))
        for i, x_t in enumerate(X_test.reshape(-1)):
            for j, x in enumerate(np.array(model.categories_).reshape(-1)):
                ans[i, j] = similarity_f(x_t, x, 3)

        numpy.testing.assert_almost_equal(X_test_enc, ans)


def test_similarity_encoder() -> None:
    categories = ["auto", "most_frequent", "k-means"]
    for category in categories:
        if category == "auto":
            _test_similarity(
                ngram_similarity,
                categories=category,
                n_prototypes=None,
            )
            _test_similarity(
                ngram_similarity,
                hashing_dim=2**16,
                categories=category,
            )
            _test_similarity(ngram_similarity, categories=category, n_prototypes=4)
        else:
            for i in range(0, 4):
                _test_similarity(
                    ngram_similarity,
                    categories=category,
                    n_prototypes=i,
                )
                _test_similarity(
                    ngram_similarity,
                    hashing_dim=2**16,
                    categories=category,
                    n_prototypes=i,
                )

    input_types = ["list", "numpy", "pandas"]
    handle_missing = ["aaa", "error", ""]
    for input_type in input_types:
        for missing in handle_missing:
            _test_missing_values(input_type, missing)
            _test_missing_values_transform(input_type, missing)


def test_kmeans_protoypes() -> None:
    X_test = np.array(["cbbba", "baaac", "accc"])
    proto = get_kmeans_prototypes(X_test, 3, sparse=True)
    assert np.array_equal(np.sort(proto), np.sort(X_test))
    X_test_2 = np.array(["aa", "bb", "cc", "bbb"])
    with pytest.warns(UserWarning, match=r"number of unique prototypes is lower "):
        get_kmeans_prototypes(X_test_2, 4)


def test_ngram_similarity_matrix() -> None:
    X1 = np.array(["cat1", "cat2", "cat3"])
    X2 = np.array(["cata1", "caat2", "ccat3"])
    sim = ngram_similarity_matrix(X1, X2, ngram_range=(2, 2), hashing_dim=5)
    assert sim.shape == (len(X1), len(X2))


def test_reproducibility() -> None:
    sim_enc = SimilarityEncoder(
        categories="k-means",
        n_prototypes=10,
        random_state=435,
    )
    X = np.array([" %s " % chr(i) for i in range(32, 127)]).reshape((-1, 1))
    prototypes = sim_enc.fit(X).categories_[0]
    for i in range(10):
        assert np.array_equal(prototypes, sim_enc.fit(X).categories_[0])


def test_fit_transform() -> None:
    X = [["foo"], ["baz"]]
    y = ["foo", "bar"]
    tr1 = SimilarityEncoder().fit_transform(X, y)
    sim = SimilarityEncoder()
    sim.fit(X, y)
    tr2 = sim.transform(X, y)
    assert np.array_equal(tr1, tr2)


def test_get_features() -> None:
    # See https://github.com/dirty-cat/skrub/issues/168
    sim_enc = SimilarityEncoder(random_state=435)
    X = np.array(["%s" % chr(i) for i in range(32, 127)]).reshape((-1, 1))
    sim_enc.fit(X)
    if parse_version(sklearn_version) < parse_version("1.0"):
        feature_names = sim_enc.get_feature_names()
    else:
        feature_names = sim_enc.get_feature_names_out()
    assert feature_names.tolist() == [
        "x0_ ",
        "x0_!",
        'x0_"',
        "x0_#",
        "x0_$",
        "x0_%",
        "x0_&",
        "x0_'",
        "x0_(",
        "x0_)",
        "x0_*",
        "x0_+",
        "x0_,",
        "x0_-",
        "x0_.",
        "x0_/",
        "x0_0",
        "x0_1",
        "x0_2",
        "x0_3",
        "x0_4",
        "x0_5",
        "x0_6",
        "x0_7",
        "x0_8",
        "x0_9",
        "x0_:",
        "x0_;",
        "x0_<",
        "x0_=",
        "x0_>",
        "x0_?",
        "x0_@",
        "x0_A",
        "x0_B",
        "x0_C",
        "x0_D",
        "x0_E",
        "x0_F",
        "x0_G",
        "x0_H",
        "x0_I",
        "x0_J",
        "x0_K",
        "x0_L",
        "x0_M",
        "x0_N",
        "x0_O",
        "x0_P",
        "x0_Q",
        "x0_R",
        "x0_S",
        "x0_T",
        "x0_U",
        "x0_V",
        "x0_W",
        "x0_X",
        "x0_Y",
        "x0_Z",
        "x0_[",
        "x0_\\",
        "x0_]",
        "x0_^",
        "x0__",
        "x0_`",
        "x0_a",
        "x0_b",
        "x0_c",
        "x0_d",
        "x0_e",
        "x0_f",
        "x0_g",
        "x0_h",
        "x0_i",
        "x0_j",
        "x0_k",
        "x0_l",
        "x0_m",
        "x0_n",
        "x0_o",
        "x0_p",
        "x0_q",
        "x0_r",
        "x0_s",
        "x0_t",
        "x0_u",
        "x0_v",
        "x0_w",
        "x0_x",
        "x0_y",
        "x0_z",
        "x0_{",
        "x0_|",
        "x0_}",
        "x0_~",
    ]


def test_check_fitted_super_vectorizer() -> None:
    """Test that calling transform before fit raises an error"""
    sim_enc = SimilarityEncoder()
    X = np.array(["%s" % chr(i) for i in range(32, 127)]).reshape((-1, 1))
    with pytest.raises(NotFittedError):
        sim_enc.transform(X)
    sim_enc.fit(X)
    sim_enc.transform(X)
