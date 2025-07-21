import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor

import skrub
from skrub import _dataframe as sbd
from skrub._data_ops import _subsampling


@pytest.mark.parametrize("as_frame", [False, True])
def test_subsample(as_frame):
    shapes = []

    def log_shape(a):
        shapes.append(a.shape)
        return a

    X_a, y_a = make_regression(random_state=0, n_features=13)
    if as_frame:
        X_a = pd.DataFrame(X_a, columns=list(map(str, range(X_a.shape[1]))))
        y_a = pd.Series(y_a, name="y")
    X = skrub.X(X_a).skb.subsample(n=15).skb.apply_func(log_shape)
    y = skrub.y(y_a).skb.subsample(n=15).skb.apply_func(log_shape)
    pred = X.skb.apply(
        DummyRegressor(strategy=skrub.choose_from(["mean", "median"])), y=y
    )
    assert shapes == [(15, 13), (15,)]
    shapes = []
    pred.skb.make_learner(fitted=True, keep_subsampling=True)
    assert shapes == [(15, 13), (15,)]
    shapes = []
    pred.skb.make_learner(fitted=True)
    assert shapes == [(100, 13), (100,)]
    shapes = []
    pred.skb.make_grid_search(fitted=True, keep_subsampling=True, cv=3)
    assert (
        shapes
        == [
            # train
            (15, 13),
            (15,),
            # test
            (34, 13),
            (34,),
            # train
            (15, 13),
            (15,),
            # test
            (33, 13),
            (33,),
            # train
            (15, 13),
            (15,),
            # test
            (33, 13),
            (33,),
        ]
        # 2 grid cells
        * 2
        # refit best model
        + [(15, 13), (15,)]
    )
    shapes = []
    pred.skb.make_grid_search(fitted=True, cv=3)
    assert (
        shapes
        == [
            # train
            (66, 13),
            (66,),
            # test
            (34, 13),
            (34,),
            # train
            (67, 13),
            (67,),
            # test
            (33, 13),
            (33,),
            # train
            (67, 13),
            (67,),
            # test
            (33, 13),
            (33,),
        ]
        # 2 grid cells
        * 2
        # refit best model
        + [(100, 13), (100,)]
    )
    with pytest.raises(ValueError):
        pred.skb.make_learner(fitted=False, keep_subsampling=True)
    with pytest.raises(ValueError):
        pred.skb.make_grid_search(fitted=False, keep_subsampling=True)


@pytest.mark.parametrize("as_frame", [False, True])
def test_how(as_frame):
    def _to_np(a):
        return a.values if as_frame else a

    X_a = np.eye(3)
    if as_frame:
        X_a = pd.DataFrame(X_a, columns=list(map(str, range(X_a.shape[1]))))

    X = skrub.X(X_a).skb.subsample(n=2)
    assert (_to_np(X.skb.eval()) == _to_np(X_a)).all()
    assert (_to_np(X.skb.eval(keep_subsampling=True)) == _to_np(X_a)[:2]).all()
    assert (_to_np(X.skb.preview()) == _to_np(X_a)[:2]).all()

    X = skrub.X(X_a).skb.subsample(n=2, how="random")
    # sampling is done differently for numpy arrays and in df.sample()
    idx = [2, 1] if as_frame else [1, 2]
    assert (_to_np(X.skb.eval()) == _to_np(X_a)).all()
    assert (_to_np(X.skb.eval(keep_subsampling=True)) == _to_np(X_a)[idx]).all()
    assert (_to_np(X.skb.preview()) == _to_np(X_a)[idx]).all()


def test_sample_errors():
    with pytest.raises(RuntimeError, match=".*`how` should be 'head' or 'random'"):
        skrub.as_data_op(np.eye(3)).skb.subsample(n=2, how="bad-how")
    with pytest.raises(RuntimeError, match=".*the input should be a dataframe"):
        skrub.as_data_op(list(range(30))).skb.subsample(n=2)


def test_should_subsample():
    should = []

    def log(a, s):
        should.append(s)
        return a

    X_a, y_a = make_regression(random_state=0, n_features=13)
    s = _subsampling.should_subsample()
    X = skrub.X(X_a).skb.apply_func(log, s)
    y = skrub.y(y_a)
    pred = X.skb.apply(
        DummyRegressor(strategy=skrub.choose_from(["mean", "median"])), y=y
    )
    assert should == [True]
    should = []
    pred.skb.make_learner(fitted=True, keep_subsampling=True)
    assert should == [True]
    should = []
    pred.skb.make_learner(fitted=True)
    assert should == [False]
    should = []
    pred.skb.make_grid_search(fitted=True, keep_subsampling=True, cv=3)
    # train/test * 3 cv folds * 2 params + refit
    assert should == [True, False, True, False, True, False] * 2 + [True]
    should = []
    pred.skb.make_grid_search(fitted=True, cv=3)
    assert should == [False] * 12 + [False]


@pytest.mark.parametrize("how", ["head", "random"])
def test_n_too_large(how, df_module):
    # subsampling is just to speed-up previews so setting n larger than a data
    # should not cause errors
    df = df_module.example_dataframe
    X = skrub.X(df).skb.subsample(n=10_000, how=how)
    assert sbd.shape(X.skb.eval())[0] == sbd.shape(df)[0]


@pytest.mark.parametrize("how", ["head", "random"])
def test_n_too_large_numpy(how):
    # subsampling is just to speed-up previews so setting n larger than a data
    # should not cause errors
    a = np.eye(3)
    X = skrub.X(a).skb.subsample(n=10_000, how=how)
    assert X.skb.eval().shape[0] == a.shape[0]


def test_uses_subsampling():
    assert not _subsampling.uses_subsampling(skrub.var("a") + 10)
    assert _subsampling.uses_subsampling(skrub.var("a").skb.subsample(n=5) + 10)


def test_subsampling_not_configured():
    with pytest.raises(ValueError, match=".*no subsampling has been configured"):
        skrub.as_data_op(np.ones(3)).skb.eval(keep_subsampling=True)
    with pytest.raises(ValueError, match=".*no subsampling has been configured"):
        skrub.as_data_op(np.ones(3)).skb.make_learner(
            fitted=True, keep_subsampling=True
        )

    # no problem if subsampling was configured
    assert (
        skrub.as_data_op(np.ones(3)).skb.subsample(n=2).skb.eval(keep_subsampling=True)
        == np.ones(2)
    ).all()
    skrub.as_data_op(np.ones(3)).skb.subsample(n=2).skb.make_learner(
        fitted=True, keep_subsampling=True
    )

    # no problem if we don't pass keep_subsampling=True
    assert (skrub.as_data_op(np.ones(3)).skb.eval() == np.ones(3)).all()
    skrub.as_data_op(np.ones(3)).skb.make_learner(fitted=True)
