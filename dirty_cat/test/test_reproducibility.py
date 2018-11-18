import numpy as np

from dirty_cat.similarity_encoder import SimilarityEncoder


def get_result(random=None):
    X = np.array(['aac', 'aaa', 'aaab', 'aaa', 'aaab', 'aaa', 'aaab', 'aaa']).reshape(-1, 1)
    X_test = np.array([['Aa', 'aAa', 'aaa', 'aaab', ' aaa  c']]).reshape(-1, 1)
    sim_enc = SimilarityEncoder(similarity='ngram', categories='auto', random_state=random)
    encoder = sim_enc.fit(X)
    transformed = encoder.transform(X_test)
    return transformed


def test_reproducibility():
    res1 = get_result(2454)
    res2 = get_result(2454)
    assert (np.array_equal(res1, res2))

    random_state = np.random.RandomState(32456)
    res1 = get_result(random_state)
    res2 = get_result(random_state)
    assert (np.array_equal(res1, res2))
