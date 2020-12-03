import time
import random
import numpy as np
import pytest

from sklearn.datasets import fetch_20newsgroups
from dirty_cat import OnlineGammaPoissonFactorization

def test_OnlineGammaPoissonFactorization(n_samples=70):
    X_txt = fetch_20newsgroups(subset='train')['data']
    X = X_txt[:n_samples]
    n_topics = 10
    for hashing in [True, False]:
        for init in ['k-means++', 'random', 'k-means']:
            for analyzer in ['word', 'char', 'char_wb']:
                for add_words in [True, False]:
                    # Test output shape
                    encoder = OnlineGammaPoissonFactorization(
                        n_topics=n_topics, hashing=hashing, init=init,
                        analyzer=analyzer, add_words=add_words,
                        random_state=42, rescale_W=True)
                    encoder.fit(X)
                    y = encoder.transform(X)
                    assert y.shape == (n_samples, n_topics), str(y.shape)
                    assert len(set(y[0])) == n_topics

                    # Test L1-norm of topics W.
                    l1_norm_W = np.abs(encoder.W_).sum(axis=1)
                    np.testing.assert_array_almost_equal(
                        l1_norm_W, np.ones(n_topics))

                    # Test same seed return the same output
                    encoder = OnlineGammaPoissonFactorization(
                        n_topics=n_topics, hashing=hashing, init=init,
                        analyzer=analyzer, add_words=add_words,
                        random_state=42)
                    encoder.fit(X)
                    y2 = encoder.transform(X)
                    np.testing.assert_array_equal(y, y2)

def test_input_type():
    # Numpy array
    X = np.array(['alice', 'bob'])
    enc = OnlineGammaPoissonFactorization(n_topics=2)
    enc.fit_transform(X)
    # List
    X = ['alice', 'bob']
    enc = OnlineGammaPoissonFactorization(n_topics=2)
    enc.fit_transform(X)

def profile_encoder(Encoder, init):
    # not an unit test

    from dirty_cat import datasets
    import pandas as pd
    employee_salaries = datasets.fetch_employee_salaries()
    data = employee_salaries['data']
    X = data['employee_position_title'].tolist()
    t0 = time.time()
    encoder = Encoder(n_topics=50, init=init)
    encoder.fit(X)
    y = encoder.transform(X)
    assert y.shape == (len(X), 50)
    eta = time.time() - t0
    return eta


if __name__ == '__main__':
    print('start test')
    test_OnlineGammaPoissonFactorization()
    print('test passed')
    print('start test_input_type')
    test_input_type()
    print('test_input_type passed')
    
    for _ in range(3):
        print('time profile_encoder(OnlineGammePoissonFactorization,',
        'init="k-means++")')
        print("{:.4} seconds".format(profile_encoder(
            OnlineGammaPoissonFactorization, init='k-means++')))
    for _ in range(3):
        print('time profile_encoder(OnlineGammePoissonFactorization,',
        'init="k-means")')
        print("{:.4} seconds".format(profile_encoder(
            OnlineGammaPoissonFactorization, init='k-means')))
    
    print('Done')
