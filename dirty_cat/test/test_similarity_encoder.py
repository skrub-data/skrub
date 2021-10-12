import numpy as np
import numpy.testing
import pytest

from dirty_cat import similarity_encoder, string_distances
from dirty_cat.similarity_encoder import get_kmeans_prototypes


def test_specifying_categories():
    # When creating a new SimilarityEncoder:
    # - if categories = 'auto', the categories are the sorted, unique training
    # set observations (for each column)
    # - if categories is a list (of lists), the categories for each column are
    # each item in the list

    # In this test, we first find the sorted, unique categories in the training
    # set, and create a SimilarityEncoder by giving it explicitly the computed
    # categories. The test consists in making sure the transformed observations
    # given by this encoder are equal to the transformed obervations in the
    # case of a SimilarityEncoder created with categories = 'auto'

    observations = [['bar'], ['foo']]
    categories = [['bar', 'foo']]

    sim_enc_with_cat = similarity_encoder.SimilarityEncoder(
        categories=categories, ngram_range=(2, 3), similarity='ngram')
    sim_enc_auto_cat = similarity_encoder.SimilarityEncoder(
        ngram_range=(2, 3), similarity='ngram')

    feature_matrix_with_cat = sim_enc_with_cat.fit_transform(observations)
    feature_matrix_auto_cat = sim_enc_auto_cat.fit_transform(observations)

    assert np.allclose(feature_matrix_auto_cat, feature_matrix_with_cat)


def test_fast_ngram_similarity():
    vocabulary = [['bar', 'foo']]
    observations = [['foo'], ['baz']]

    sim_enc = similarity_encoder.SimilarityEncoder(
        similarity='ngram', ngram_range=(2, 2), categories=vocabulary)

    sim_enc.fit(observations)
    feature_matrix = sim_enc.transform(observations, fast=False)
    feature_matrix_fast = sim_enc.transform(observations, fast=True)

    assert np.allclose(feature_matrix, feature_matrix_fast)


def _test_missing_values(input_type, missing):
    observations = [['a', 'b'], ['b', 'a'], ['b', np.nan],
                    ['a', 'c'], [np.nan, 'a']]
    encoded = np.array([[0., 1., 0., 0., 0., 1., 0.],
                        [0., 0., 1., 0., 1., 0., 0.],
                        [0., 0., 1., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 0., 1.],
                        [0., 0., 0., 0., 1., 0., 0.]])

    if input_type == 'numpy':
        observations = np.array(observations, dtype=object)
    elif input_type == 'pandas':
        pd = pytest.importorskip("pandas")
        observations = pd.DataFrame(observations)

    sim_enc = similarity_encoder.SimilarityEncoder(handle_missing=missing)
    if missing == 'error':
        with pytest.raises(ValueError, match=r"Found missing values in input "
                           "data; set handle_missing='' to encode "
                           "with missing values"):
            sim_enc.fit_transform(observations)
    elif missing == '':
        ans = sim_enc.fit_transform(observations)
        assert np.allclose(encoded, ans)
    else:
        with pytest.raises(ValueError, match=r"handle_missing"
                           " should be either 'error' or ''"):
            sim_enc.fit_transform(observations)
        return


def _test_missing_values_transform(input_type, missing):
    observations = [['a', 'b'], ['b', 'a'], ['b', 'b'],
                    ['a', 'c'], ['c', 'a']]
    test_observations = [['a', 'b'], ['b', 'a'], ['b', np.nan],
                         ['a', 'c'], [np.nan, 'a']]
    encoded = np.array([[1., 0., 0., 0., 1., 0.],
                        [0., 1., 0., 1., 0., 0.],
                        [0., 1., 0., 0., 0., 0.],
                        [1., 0., 0., 0., 0., 1.],
                        [0., 0., 0., 1., 0., 0.]])

    if input_type == 'numpy':
        test_observations = np.array(test_observations, dtype=object)
    elif input_type == 'pandas':
        pd = pytest.importorskip("pandas")
        test_observations = pd.DataFrame(test_observations)

    sim_enc = similarity_encoder.SimilarityEncoder(handle_missing=missing)
    if missing == 'error':
        sim_enc.fit_transform(observations)
        with pytest.raises(ValueError, match=r"Found missing values in input "
                           "data; set handle_missing='' to encode "
                           "with missing values"):
            sim_enc.transform(test_observations)
    elif missing == '':
        sim_enc.fit_transform(observations)
        ans = sim_enc.transform(test_observations)
        assert np.allclose(encoded, ans)


def _test_similarity(similarity, similarity_f, hashing_dim=None, categories='auto', n_prototypes=None):
    if n_prototypes is None:
        X = np.array(['aa', 'aaa', 'aaab']).reshape(-1, 1)
        X_test = np.array([['Aa', 'aAa', 'aaa', 'aaab', ' aaa  c']]).reshape(-1, 1)

        model = similarity_encoder.SimilarityEncoder(
            similarity=similarity, hashing_dim=hashing_dim, categories=categories,
            n_prototypes=n_prototypes)

        if similarity == 'ngram':
            model.ngram_range = (3, 3)

        encoder = model.fit(X).transform(X_test)

        ans = np.zeros((len(X_test), len(X)))
        for i, x_t in enumerate(X_test.reshape(-1)):
            for j, x in enumerate(X.reshape(-1)):
                if similarity == 'ngram':
                    ans[i, j] = similarity_f(x_t, x, 3)
                else:
                    ans[i, j] = similarity_f(x_t, x)
        numpy.testing.assert_almost_equal(encoder, ans)
    else:
        X = np.array(
            ['aac', 'aaa', 'aaab', 'aaa', 'aaab', 'aaa', 'aaab', 'aaa']
        ).reshape(-1, 1)
        X_test = np.array([['Aa', 'aAa', 'aaa', 'aaab', ' aaa  c']]
                          ).reshape(-1, 1)

        try:
            model = similarity_encoder.SimilarityEncoder(
                similarity=similarity, hashing_dim=hashing_dim, categories=categories,
                n_prototypes=n_prototypes, random_state=42)
        except ValueError as e:
            assert (e.__str__() == 'n_prototypes expected None or a positive non null integer')
            return

        if similarity == 'ngram':
            model.ngram_range = (3, 3)

        encoder = model.fit(X).transform(X_test)
        if n_prototypes == 1:
            assert (model.categories_ == ['aaa'])
        elif n_prototypes == 2:
            a = [np.array(['aaa', 'aaab'], dtype='<U4')]
            assert (np.array_equal(a, model.categories_))
        elif n_prototypes == 3:
            a = [np.array(['aaa', 'aaab', 'aac'], dtype='<U4')]
            assert (np.array_equal(a, model.categories_))

        ans = np.zeros((len(X_test),
                        len(np.array(model.categories_).reshape(-1))))
        for i, x_t in enumerate(X_test.reshape(-1)):
            for j, x in enumerate(np.array(model.categories_).reshape(-1)):
                if similarity == 'ngram':
                    ans[i, j] = similarity_f(x_t, x, 3)
                else:
                    ans[i, j] = similarity_f(x_t, x)

        numpy.testing.assert_almost_equal(encoder, ans)


def test_similarity_encoder():
    categories = ['auto', 'most_frequent', 'k-means']
    for category in categories:
        if category == 'auto':
            _test_similarity('levenshtein-ratio',
                             string_distances.levenshtein_ratio,
                             categories=category,
                             n_prototypes=None)
            _test_similarity('jaro-winkler', string_distances.jaro_winkler,
                             categories=category, n_prototypes=None)
            _test_similarity('jaro', string_distances.jaro,
                             categories=category, n_prototypes=None)
            _test_similarity('ngram', string_distances.ngram_similarity,
                             categories=category, n_prototypes=None)
            _test_similarity('ngram', string_distances.ngram_similarity,
                             hashing_dim=2 ** 16, categories=category)
        else:
            for i in range(1, 4):
                _test_similarity('levenshtein-ratio',
                                 string_distances.levenshtein_ratio,
                                 categories=category,
                                 n_prototypes=i)
                _test_similarity('jaro-winkler', string_distances.jaro_winkler,
                                 categories=category, n_prototypes=i)
                _test_similarity('jaro', string_distances.jaro,
                                 categories=category, n_prototypes=i)
                _test_similarity('ngram', string_distances.ngram_similarity,
                                 categories=category, n_prototypes=i)
                _test_similarity('ngram', string_distances.ngram_similarity,
                                 hashing_dim=2 ** 16, categories=category,
                                 n_prototypes=i)

    input_types = ['list', 'numpy', 'pandas']
    handle_missing = ['aaa', 'error', '']
    for input_type in input_types:
        for missing in handle_missing:
            _test_missing_values(input_type, missing)
            _test_missing_values_transform(input_type, missing)


def test_kmeans_protoypes():
    X_test = np.array(['cbbba', 'baaac', 'accc'])
    proto = get_kmeans_prototypes(X_test, 3)
    assert np.array_equal(np.sort(proto), np.sort(X_test))


def test_reproducibility():
    sim_enc = similarity_encoder.SimilarityEncoder(categories='k-means', n_prototypes=10, random_state=435)
    X = np.array([' %s ' % chr(i) for i in range(32, 127)]).reshape((-1, 1))
    prototypes = sim_enc.fit(X).categories_[0]
    for i in range(10):
        assert (np.array_equal(prototypes, sim_enc.fit(X).categories_[0]))


def test_get_features():
    # See https://github.com/dirty-cat/dirty_cat/issues/168
    sim_enc = similarity_encoder.SimilarityEncoder(random_state=435)
    X = np.array(['%s' % chr(i) for i in range(32, 127)]).reshape((-1, 1))
    sim_enc.fit(X)
    feature_names = sim_enc.get_feature_names()
    assert feature_names.tolist() == [
        'x0_ ', 'x0_!', 'x0_"', 'x0_#', 'x0_$', 'x0_%', 'x0_&', "x0_'", 'x0_(',
        'x0_)', 'x0_*', 'x0_+', 'x0_,', 'x0_-', 'x0_.', 'x0_/', 'x0_0', 'x0_1',
        'x0_2', 'x0_3', 'x0_4', 'x0_5', 'x0_6', 'x0_7', 'x0_8', 'x0_9', 'x0_:',
        'x0_;', 'x0_<', 'x0_=', 'x0_>', 'x0_?', 'x0_@', 'x0_A', 'x0_B', 'x0_C',
        'x0_D', 'x0_E', 'x0_F', 'x0_G', 'x0_H', 'x0_I', 'x0_J', 'x0_K', 'x0_L',
        'x0_M', 'x0_N', 'x0_O', 'x0_P', 'x0_Q', 'x0_R', 'x0_S', 'x0_T', 'x0_U',
        'x0_V', 'x0_W', 'x0_X', 'x0_Y', 'x0_Z', 'x0_[', 'x0_\\', 'x0_]', 'x0_^',
        'x0__', 'x0_`', 'x0_a', 'x0_b', 'x0_c', 'x0_d', 'x0_e', 'x0_f', 'x0_g',
        'x0_h', 'x0_i', 'x0_j', 'x0_k', 'x0_l', 'x0_m', 'x0_n', 'x0_o', 'x0_p',
        'x0_q', 'x0_r', 'x0_s', 'x0_t', 'x0_u', 'x0_v', 'x0_w', 'x0_x', 'x0_y',
        'x0_z', 'x0_{', 'x0_|', 'x0_}', 'x0_~'
    ]

