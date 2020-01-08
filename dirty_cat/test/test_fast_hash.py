from dirty_cat.fast_hash import ngram_min_hash


def test_fast_hash():

    from sklearn import datasets
    data = datasets.fetch_20newsgroups()
    a = data.data[0]

    min_hash = ngram_min_hash(a, seed=0)
    min_hash2 = ngram_min_hash(a, seed=0)
    assert min_hash == min_hash2

    list_min_hash = [ngram_min_hash(a, seed=seed) for seed in range(50)]
    assert len(set(list_min_hash)) > 45, 'Too many hash collisions'

    min_hash4 = ngram_min_hash(a, seed=0, return_minmax=True)
    assert len(min_hash4) == 2


if __name__ == '__main__':
    test_fast_hash()
