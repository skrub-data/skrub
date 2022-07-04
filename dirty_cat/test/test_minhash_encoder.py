import time
import random
from string import ascii_lowercase
import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import fetch_20newsgroups
from dirty_cat import MinHashEncoder

def test_MinHashEncoder(n_sample=70, minmax_hash=False):
    X_txt = fetch_20newsgroups(subset='train')['data']
    X = np.array(X_txt[:n_sample])[:,None]

    for minmax_hash in [True, False]:
        for hashing in ['fast', 'murmur']:

            if minmax_hash and hashing == 'murmur':
                pass # not implemented

            # Test output shape
            encoder = MinHashEncoder(n_components=50, hashing=hashing)
            encoder.fit(X)
            y = encoder.transform(X)
            assert y.shape == (n_sample, 50), str(y.shape)
            assert len(set(y[0])) == 50

            # Test same seed return the same output
            encoder = MinHashEncoder(50, hashing=hashing)
            encoder.fit(X)
            y2 = encoder.transform(X)
            np.testing.assert_array_equal(y, y2)

            # Test min property
            if not minmax_hash:
                X_substring = [x[:x.find(' ')] for x in X[:,0]]
                X_substring = np.array(X_substring)[:,None]
                encoder = MinHashEncoder(50, hashing=hashing)
                encoder.fit(X_substring)
                y_substring = encoder.transform(X_substring)
                np.testing.assert_array_less(y - y_substring, 0.0001)

def test_multiple_columns():
    """ This test is intented to verify that fitting multiple columns
        with the MinHashEncoder will not produce an error, and will 
        encode the column independently """
    X = pd.DataFrame([('bird', 'parrot'),
                      ('bird', 'nightingale'),
                      ('mammal', 'monkey'),
                      ('mammal', np.nan)],
                      columns=('class', 'type'))
    X1 = X[['class']]
    X2 = X[['type']]
    fit1 = MinHashEncoder(n_components=30).fit_transform(X1)
    fit2 = MinHashEncoder(n_components=30).fit_transform(X2)
    fit = MinHashEncoder(n_components=30).fit_transform(X)
    assert np.array_equal(np.array([fit[:, :30], fit[:, 30:60]]), np.array([fit1, fit2]))

def test_input_type():
    # Numpy array
    X = np.array(['alice', 'bob'])[:,None]
    enc = MinHashEncoder(n_components=2)
    enc.fit_transform(X)
    # List
    X = [['alice'], ['bob']]
    enc = MinHashEncoder(n_components=2)
    enc.fit_transform(X)

def test_get_unique_ngrams():
    string = 'test'
    true_ngrams = {
        (' ','t'), ('t','e'), ('e','s'), ('s', 't'),
        ('t',' '), (' ','t','e'), ('t','e','s'),
        ('e','s','t'), ('s','t',' '), (' ','t','e','s'),
        ('t','e','s','t'), ('e','s','t',' ')}
    ngram_range = (2,4)
    enc = MinHashEncoder(n_components=2)
    ngrams = enc.get_unique_ngrams(string, ngram_range)
    assert ngrams == true_ngrams


def profile_encoder(encoder, hashing='fast', minmax_hash=False):
    # not an unit test

    from dirty_cat.datasets import fetch_employee_salaries
    employee_salaries = fetch_employee_salaries()
    df = employee_salaries.X
    X = df[["employee_position_title"]]
    t0 = time.time()
    enc = encoder(n_components=50, hashing=hashing, minmax_hash=minmax_hash)
    enc.fit(X)
    y = enc.transform(X)
    assert y.shape == (len(X), 50)
    eta = time.time() - t0
    return eta


@pytest.mark.parametrize("input_type, missing, hashing", [
    ['numpy', 'error', 'fast'],
    ['pandas', 'zero_impute', 'murmur'],
    ['numpy', 'zero_impute', 'fast']])
def test_missing_values(input_type, missing, hashing):
    X = ['Red',
         np.nan,
         'green',
         'blue',
         'green',
         'green',
         'blue',
         float('nan')]
    n = 3
    z = np.zeros(n)

    if input_type == 'numpy':
        X = np.array(X, dtype=object)[:,None]
    elif input_type == 'pandas':
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame(X)

    encoder = MinHashEncoder(n_components=n, hashing=hashing,
                             minmax_hash=False, handle_missing=missing)
    if missing == 'error':
        encoder.fit(X)
        if input_type in ['numpy', 'pandas']:
            with pytest.raises(ValueError, match=r"missing"
                               " values in input"):
                encoder.transform(X)
    elif missing == 'zero_impute':
        encoder.fit(X)
        y = encoder.transform(X)
        if input_type == 'list':
            assert np.allclose(y[1], y[-1])
        else:
            assert np.array_equal(y[1], z)
            assert np.array_equal(y[-1], z)
    else:
        with pytest.raises(ValueError, match=r"handle_missing"
                           " should be either 'error' or 'zero_impute'"):
            encoder.fit_transform(X)
    return


def test_cache_overflow():
    # Regression test for cache overflow resulting in -1s in encoding
    def get_random_string(length):
        letters = ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str

    encoder = MinHashEncoder(n_components=3)
    capacity = encoder._capacity
    raw_data = [get_random_string(10) for x in range(capacity + 1)]
    raw_data = np.array(raw_data)[:,None]
    y = encoder.fit_transform(raw_data)

    assert len(y[y == -1.0]) == 0


if __name__ == '__main__':
    print('start test')
    test_MinHashEncoder()
    print('test passed')
    
    print('start test')
    test_multiple_columns()
    print('multiple columns encoding test passed')

    for _ in range(3):
        print('time profile_encoder(MinHashEncoder, hashing=fast)')
        print("{:.4} seconds".format(
            profile_encoder(MinHashEncoder, hashing='fast')))
    for _ in range(3):
        print('time profile_encoder(MinHashEncoder, hashing=fast) with minmax')
        print("{:.4} seconds".format(profile_encoder(MinHashEncoder,
                                     hashing='fast', minmax_hash=True)))
    print('time profile_encoder(MinHashEncoder, hashing=murmur)')
    print("{:.4} seconds".format(
        profile_encoder(MinHashEncoder, hashing='murmur')))

    print('Done')
