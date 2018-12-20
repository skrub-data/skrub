import numpy as np
import numpy.testing
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
