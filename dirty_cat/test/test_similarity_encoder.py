import numpy as np

from dirty_cat import similarity_encoder, string_distances


def _test_similarity(similarity, similarity_f, hashing_dim=None, categories='auto', n_prototypes=None):
    X = np.array(['aa', 'aaa', 'aaab']).reshape(-1, 1)
    X_test = np.array([['Aa', 'aAa', 'aaa', 'aaab', ' aaa  c']]).reshape(-1, 1)

    model = similarity_encoder.SimilarityEncoder(
        similarity=similarity, handle_unknown='ignore',
        hashing_dim=hashing_dim, categories=categories,
        n_prototypes=n_prototypes)

    encoder = model.fit(X).transform(X_test)

    ans = np.zeros((len(X_test), len(X)))
    for i, x_t in enumerate(X_test.reshape(-1)):
        for j, x in enumerate(X.reshape(-1)):
            if similarity == 'ngram':
                ans[i, j] = similarity_f(x_t, x, 3)
            else:
                ans[i, j] = similarity_f(x_t, x)
    assert np.array_equal(encoder, ans)


def test_similarity_encoder():

    categories = ['auto', 'most_frequent']

    for category in categories:
        if category=='auto':
            _test_similarity('levenshtein-ratio', string_distances.levenshtein_ratio, categories=category, n_prototypes=None)
            _test_similarity('jaro-winkler', string_distances.jaro_winkler, categories=category, n_prototypes=None)
            _test_similarity('jaro', string_distances.jaro, categories=category, n_prototypes=None)
            _test_similarity('ngram', string_distances.ngram_similarity, categories=category, n_prototypes=None)
            _test_similarity('ngram', string_distances.ngram_similarity, hashing_dim=2**16, categories=category, n_prototypes=None)
        else:
            _test_similarity('levenshtein-ratio', string_distances.levenshtein_ratio, categories=category, n_prototypes=1)
            _test_similarity('jaro-winkler', string_distances.jaro_winkler, categories=category, n_prototypes=1)
            _test_similarity('jaro', string_distances.jaro, categories=category, n_prototypes=1)
            _test_similarity('ngram', string_distances.ngram_similarity, categories=category, n_prototypes=1)
            _test_similarity('ngram', string_distances.ngram_similarity, hashing_dim=2**16, categories=category, n_prototypes=1)
