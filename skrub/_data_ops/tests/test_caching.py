import warnings
from collections import defaultdict

import pytest
from sklearn.base import BaseEstimator, TransformerMixin

import skrub


class DummyTransformer(TransformerMixin, BaseEstimator):
    n_calls = defaultdict(int)

    def __init__(self, add=1):
        self.add = add

    def fit(self, X, y=None):
        self.n_calls["fit"] += 1
        return self

    def fit_transform(self, X, y=None):
        self.n_calls["fit_transform"] += 1
        return X + self.add

    def transform(self, X):
        self.n_calls["transform"] += 1
        return X + self.add

    @staticmethod
    def reset():
        DummyTransformer.n_calls = defaultdict(int)


def f(x, add=10):
    f.n_calls += 1
    return x + add


f.n_calls = 0


@pytest.fixture(autouse=True)
def reset_counts():
    f.n_calls = 0
    DummyTransformer.reset()


@pytest.mark.parametrize("with_cache", (False, True))
def test_caching(with_cache, tmp_path):
    if with_cache:
        skrub.set_config(cache=tmp_path)
    data_op = (
        skrub.var("x")
        .skb.apply(DummyTransformer(1))
        .skb.apply_func(f, 10)
        .skb.apply(DummyTransformer(2), no_cache=True)
        .skb.apply_func(f, 20, no_cache=True)
    )
    learner = data_op.skb.make_learner()
    out = learner.fit_transform({"x": 1})
    assert out == 34
    assert DummyTransformer.n_calls == {"fit_transform": 2}
    assert f.n_calls == 2
    out = learner.fit_transform({"x": 1})
    assert out == 34
    assert DummyTransformer.n_calls == {"fit_transform": 3 if with_cache else 4}
    assert f.n_calls == 3 if with_cache else 4
    out = learner.fit_transform({"x": 2})
    assert out == 35
    assert DummyTransformer.n_calls == {"fit_transform": 5 if with_cache else 6}
    assert f.n_calls == 5 if with_cache else 6


@pytest.mark.parametrize(
    "apply_func, deferred, n_calls",
    [(False, False, 1), (False, True, 2), (True, False, 2), (True, True, 2)],
)
def test_apply_deferred_func(apply_func, deferred, n_calls, tmp_path):
    with warnings.catch_warnings():
        # warning should not apply_func(deferred(...))
        warnings.simplefilter("ignore")
        skrub.set_config(cache=tmp_path)
        data_op = skrub.var("x").skb.apply_func(
            skrub.deferred(f, no_cache=deferred), no_cache=apply_func
        )
        data_op.skb.eval({"x": 0})
        data_op.skb.eval({"x": 0})
        assert f.n_calls == n_calls
