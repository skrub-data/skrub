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

    def fit_transform(self, X, y=None, **kwargs):
        self.n_calls["fit_transform"] += 1
        return X + self.add

    def transform(self, X, **kwargs):
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
    assert f.n_calls == (3 if with_cache else 4)
    out = learner.fit_transform({"x": 2})
    assert out == 35
    assert DummyTransformer.n_calls == {"fit_transform": 5 if with_cache else 6}
    assert f.n_calls == (5 if with_cache else 6)

    f.n_calls = 0
    DummyTransformer.reset()

    out = learner.transform({"x": 2})
    assert DummyTransformer.n_calls["transform"] == 2
    assert f.n_calls == (1 if with_cache else 2)

    out = learner.transform({"x": 2})
    assert DummyTransformer.n_calls["transform"] == (3 if with_cache else 4)
    assert f.n_calls == (2 if with_cache else 4)

    skrub.set_config(cache=False)
    out = learner.transform({"x": 2})
    assert DummyTransformer.n_calls["transform"] == (5 if with_cache else 6)
    assert f.n_calls == (4 if with_cache else 6)


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


def test_pickling_error(tmp_path):
    skrub.set_config(cache=tmp_path)
    a = skrub.var("a", 0, becomes_default=True)
    data_op = (
        skrub.as_data_op(3)
        .skb.apply(
            DummyTransformer(), fit_transform_kwargs={"a": a}, transform_kwargs={"a": a}
        )
        .skb.apply_func(lambda x: x)
    )
    assert data_op.skb.eval() == 4
    assert data_op.skb.eval({"a": lambda: None}) == 4
    assert data_op.skb.make_learner(fitted=True).transform({"a": lambda: None}) == 4


def test_cache_reuse_across_data_ops(tmp_path):
    # cache reused with different estimator objects in different graphs
    skrub.set_config(cache=tmp_path)
    skrub.var("x").skb.apply(DummyTransformer(1)).skb.apply_func(f, 10).skb.eval(
        {"x": 1}
    )
    assert DummyTransformer.n_calls == {"fit_transform": 1}
    skrub.var("x").skb.apply(DummyTransformer(1)).skb.apply_func(f, 20).skb.eval(
        {"x": 1}
    )
    assert DummyTransformer.n_calls == {"fit_transform": 1}


def test_deferred_no_cache_decorator(tmp_path):
    # no_cache is passed in separate call: deferred(no_cache=True)(f) rather
    # than deferred(f, no_cache=True)
    skrub.set_config(cache=tmp_path)
    g = skrub.deferred(no_cache=True)(f)
    g(skrub.var("x")).skb.eval({"x": 0})
    g(skrub.var("x")).skb.eval({"x": 0})
    assert f.n_calls == 2


def test_memory_cache():
    mem = skrub._data_ops._caching.Memory()
    assert not mem.has_memory()
    assert mem.cache(f) is f
