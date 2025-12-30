import copy
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

import skrub
from skrub import ApplyToCols
from skrub import selectors as s
from skrub._data_ops import _data_ops
from skrub._utils import PassThrough


def test_simple_data_op():
    a = skrub.var("a")
    b = skrub.var("b")
    c = a + b
    d = c * c
    assert d.skb.eval({"a": 2, "b": 3}) == 25


def test_containers():
    a = skrub.var("a")
    b = skrub.var("b")
    c = skrub.var("c")
    d = c + [{"k": (a, b)}]
    assert d.skb.eval({"a": 2, "b": 3, "c": []}) == [{"k": (2, 3)}]


def test_slice():
    a = skrub.var("a")
    b = skrub.var("b")
    c = a[:b]
    assert c.skb.eval({"a": list(range(10)), "b": 3}) == [0, 1, 2]
    assert c.skb.eval({"a": list(range(10, 20)), "b": 5}) == [10, 11, 12, 13, 14]


def test_estimator():
    c = skrub.var("c")
    e = skrub.as_data_op(LogisticRegression(C=c))
    assert e.skb.eval({"c": 0.001}).C == 0.001


def test_environement_with_values():
    a = skrub.var("a", "hello")
    b = skrub.var("b", "world")
    c = skrub.var("c", ", ")
    d = c.join([a, b]).skb.set_name("d")
    e = skrub.var("e", "!")
    f = d + e

    # if we do not provide an environment, or it contains none of the variable
    # names, the variable `value` are used.
    assert f.skb.eval() == "hello, world!"
    assert f.skb.eval({}) == "hello, world!"
    # we can still inject values for internal nodes or choices in this setting
    assert f.skb.eval({"d": "goodbye"}) == "goodbye!"

    # however if we provide a binding for any of the variables we must do it
    # for all the variables actually used, `value` is not considered for any
    # variable.
    with pytest.raises((KeyError, RuntimeError)):
        # Note: errors and messages are checked in detail in `test_errors.py`,
        # not here.
        f.skb.eval({"a": "welcome"})
    assert (
        f.skb.eval({"a": "welcome", "b": "here", "c": " ", "e": " :)"})
        == "welcome here :)"
    )

    # it is fine for a variable that is not actually used to be missing from
    # the environment. Here we override the value of 'd' so a, b, c are not
    # needed to compute it
    with pytest.raises((KeyError, RuntimeError)):
        f.skb.eval({"e": "3"})
    assert f.skb.eval({"d": "12", "e": "3"}) == "123"


def test_environment_no_values():
    a = skrub.var("a")
    b = skrub.var("b")
    c = (a + b).skb.set_name("c")
    d = skrub.var("d")
    e = c * d
    assert e.skb.eval({"a": 10, "b": 5, "d": 2}) == 30
    assert e.skb.eval({"a": 10, "b": 5, "d": 2, "c": 3}) == 6
    # Here a, b are missing from the environment but not needed because we
    # override the computation of c
    assert e.skb.eval({"d": 2, "c": 3}) == 6


def test_environment_wrong_values():
    a = skrub.var(name="a", value=[1, 2, 3])
    # Testing data_op as value
    with pytest.raises(
        TypeError, match=r".*`value` of a `skrub.var\(\)` must not contain a skrub.*"
    ):
        skrub.var(name="wrongvar", value=a)

    with pytest.raises(
        TypeError, match=r".*`value` of a `skrub.var\(\)` must not contain a skrub.*"
    ):
        skrub.X(value=a)

    with pytest.raises(
        TypeError, match=r".*`value` of a `skrub.var\(\)` must not contain a skrub.*"
    ):
        skrub.y(value=a)

    # Testing choice as value
    with pytest.raises(
        TypeError, match=r".*`value` of a `skrub.var\(\)` must not contain a skrub.*"
    ):
        skrub.var("wrongvar", skrub.choose_bool())

    with pytest.raises(
        TypeError, match=r".*`value` of a `skrub.var\(\)` must not contain a skrub.*"
    ):
        skrub.X(skrub.choose_bool())

    with pytest.raises(
        TypeError, match=r".*`value` of a `skrub.var\(\)` must not contain a skrub.*"
    ):
        skrub.y(skrub.choose_bool())


def test_choice_in_environment():
    a = skrub.var("a", 100)
    b = skrub.var("b", 10)
    c = skrub.choose_from([1, 2], name="c")
    d = a + b + c
    assert d.skb.eval() == 111
    assert d.skb.eval({}) == 111
    # we can provide a value for a choice without needing to provide bindings
    # for all variables. Note it does not need to be one of the choice's
    # outcomes.
    assert d.skb.eval({"c": 3}) == 113
    with pytest.raises((KeyError, RuntimeError)):
        d.skb.eval({"c": 3, "b": 20})
    assert d.skb.eval({"c": 3, "b": 20, "a": 400}) == 423


def test_if_else():
    a = skrub.var("a")
    b = skrub.var("b")
    c = skrub.var("c")
    d = a.skb.if_else(b, c)
    assert d.skb.eval({"a": True, "b": 10, "c": -10}) == 10
    # The alternative branch is not evaluated (otherwise we would get a
    # KeyError for 'c' here) (this is the main reason why we have `if_else`):
    assert d.skb.eval({"a": True, "b": 10}) == 10
    assert d.skb.eval({"a": False, "c": 3}) == 3


def test_match():
    a = skrub.var("a")
    b = skrub.var("b")
    c = skrub.var("c")
    d = a.skb.match({"left": b, "right": c})
    assert d.skb.eval({"a": "left", "b": 10, "c": -10}) == 10
    # The alternative branches are not evaluated (otherwise we would get a
    # KeyError for 'c' here) (this is the main reason why we have `match`):
    assert d.skb.eval({"a": "left", "b": 10}) == 10
    assert d.skb.eval({"a": "right", "c": 3}) == 3

    # if there is no match we get KeyError:
    with pytest.raises((KeyError, RuntimeError)):
        d.skb.eval({"a": "missing key", "b": 0, "c": 0})

    # unless we provide a default:
    e = skrub.var("e")
    d = a.skb.match({"left": b, "right": c}, default=e)
    assert d.skb.eval({"a": "missing key", "b": 0, "c": 0, "e": 4}) == 4

    # with the default as well, only the used branch is evaluated
    assert d.skb.eval({"a": "missing key", "e": 4}) == 4
    # no value for 'e'
    assert d.skb.eval({"a": "left", "b": 10}) == 10


def test_predictor_as_transformer():
    pred = skrub.X().skb.apply(LogisticRegression(), y=skrub.y()) * 7
    assert pred.skb.eval({"X": [[10], [-10]], "y": [0, 1]})[1] == 7.0

    X = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    pred = skrub.X().skb.apply(DummyRegressor(), y=skrub.y())
    learner = pred.skb.make_learner()
    expected = pd.DataFrame({"a": [2.0, 2.0, 2.0], "b": [20.0, 20.0, 20.0]})
    assert_frame_equal(learner.fit_transform({"X": X, "y": X}), expected)
    assert_frame_equal(learner.transform({"X": X, "y": X}), expected)


def test_predictor_outputs():
    X, y = make_classification(n_samples=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10)
    seen_modes = []

    class LogReg(LogisticRegression):
        n_predict_calls = 0

        def predict(self, X):
            LogReg.n_predict_calls += 1
            return super().predict(X)

    def check_output(output, mode):
        seen_modes.append(mode)
        if mode == "fit":
            assert isinstance(output, LogisticRegression)
            assert hasattr(output, "coef_")
        if mode == "predict":
            assert isinstance(output, np.ndarray)
            assert output.shape == (10,)
        if mode == "score":
            assert isinstance(output, float)
            assert 0.0 <= output <= 1.0
        return output

    learner = (
        skrub.X()
        .skb.apply(LogReg(), y=skrub.y())
        .skb.apply_func(check_output, skrub.eval_mode())
        .skb.make_learner()
    )
    assert learner.fit({"X": X_train, "y": y_train}) is learner
    assert LogReg.n_predict_calls == 0
    pred = learner.predict({"X": X_test, "y": y_test})
    assert pred.shape == (10,)
    assert LogReg.n_predict_calls == 1
    score = learner.score({"X": X_test, "y": y_test})
    assert 0.0 <= score <= 1.0
    assert LogReg.n_predict_calls == 2
    assert seen_modes == ["fit", "predict", "score"]


def test_predictor_output_formatting():
    X, y = make_regression(n_samples=20, n_targets=3)
    X_df = pd.DataFrame(X, columns=[f"x_{i}" for i in range(X.shape[1])])
    y_df = pd.DataFrame(y, columns=[f"y_{i}" for i in range(y.shape[1])])
    pred = skrub.X().skb.apply(DummyRegressor(), y=skrub.y())

    # When X or y is not a column or dataframe, no post-processing
    out = pred.skb.eval({"X": X, "y": y})
    assert isinstance(out, np.ndarray)
    out = pred.skb.eval({"X": X_df, "y": y})
    assert isinstance(out, np.ndarray)
    out = pred.skb.eval({"X": X, "y": y_df})
    assert isinstance(out, np.ndarray)

    # When ytrain was a dataframe, we get a dataframe
    out = pred.skb.eval({"X": X_df, "y": y_df})
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(y_df.columns)

    # Also true when it is a dataframe with only 1 column
    out = pred.skb.eval({"X": X_df, "y": y_df["y_0"]})
    assert isinstance(out, pd.Series)
    assert out.name == "y_0"

    # When ytrain was a column, we get a column
    out = pred.skb.eval({"X": X_df, "y": y_df[["y_0"]]})
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["y_0"]

    # When the estimator returns something else than a numpy array, no post-processing
    class DfPredictor(DummyRegressor):
        def predict(self, X):
            y = super().predict(X)
            return pd.DataFrame(y, columns=[f"dfpred_{i}" for i in range(y.shape[1])])

    pred = skrub.X().skb.apply(DfPredictor(), y=skrub.y())
    out = pred.skb.eval({"X": X_df, "y": y_df})
    assert list(out.columns) == ["dfpred_0", "dfpred_1", "dfpred_2"]

    # When the prediction does not have the expected shape, no post-processing
    class WrongShape(DummyRegressor):
        def predict(self, X):
            return super().predict(X)[:, :-1]

    pred = skrub.X().skb.apply(WrongShape(), y=skrub.y())
    out = pred.skb.eval({"X": X_df, "y": y_df})
    assert isinstance(out, np.ndarray)


def test_make_learner():
    learner = (skrub.var("a", 0) + skrub.var("b", 1)).skb.make_learner()
    assert learner.fit_transform({"a": 10, "b": 2}) == 12
    assert learner.transform({"a": 100, "b": 30}) == 130


def test_make_learner_choose_options():
    dop = skrub.as_data_op(
        (skrub.choose_from(("a", "b", "c")), skrub.choose_int(10, 15, default=11))
    )
    default = dop.skb.make_learner()
    assert default.fit_transform({}) == ("a", 11)

    def params(learner):
        return {k: v for k, v in learner.get_params().items() if k != "data_op"}

    assert params(default) == {"data_op__0": None, "data_op__1": None}
    for choose in ["random", "random ( ) "]:
        all_cf, all_ci = set(), set()
        for _ in range(200):
            random = dop.skb.make_learner(choose=choose)
            cf, ci = random.fit_transform({})
            all_cf.add(cf)
            all_ci.add(ci)
        assert all_cf == {"a", "b", "c"}
        assert all_ci == set(range(10, 16))
    # testing numpy RandomState and random state with seed 1
    # also testing random spacing in the seed number
    for choose in [check_random_state(1), "random (1 ) "]:
        seeded = dop.skb.make_learner(choose=choose)
        assert seeded.fit_transform({}) == ("b", 15)
        assert params(seeded) == {"data_op__0": 1, "data_op__1": 15}

    with pytest.raises(ValueError, match=r"`choose` should be"):
        dop.skb.make_learner(choose="randomized")


def test_make_learner_choices_before_X():
    # check that choices that are children of X (same for y) are forced to take
    # their default value
    msg = "The following choices are used in the construction of X or y.*choice_clamped"
    with pytest.warns(UserWarning, match=msg):
        X = (
            skrub.as_data_op(skrub.choose_from(("a", "b", "c"), name="choice_clamped"))
            + "x"
        ).skb.mark_as_X()
    with pytest.warns(UserWarning, match=msg):
        out = X + skrub.choose_from(("a", "b", "c"), name="choice_free")
    all_results = set()
    with pytest.warns(UserWarning, match=msg):
        for _ in range(30):
            all_results.add(out.skb.make_learner(choose="random").fit_transform({}))
    assert all_results == {"axa", "axb", "axc"}


def test_make_learner_choose_optuna_trial():
    optuna = pytest.importorskip("optuna")

    def params(learner):
        return {k: v for k, v in learner.get_params().items() if k != "data_op"}

    dop = skrub.as_data_op(
        (skrub.choose_from(("a", "b", "c")), skrub.choose_int(10, 15, default=11))
    )

    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(sampler=sampler)

    # The choices are returned by the Trial returned by study.ask(). Those are
    # deterministic as the sampler is seeded.
    learner = dop.skb.make_learner(choose=study.ask())
    assert learner.fit_transform({}) == ("b", 13)
    assert params(learner) == {"data_op__0": 1, "data_op__1": 13}

    learner = dop.skb.make_learner(choose=study.ask())
    assert learner.fit_transform({}) == ("b", 15)
    assert params(learner) == {"data_op__0": 1, "data_op__1": 15}

    learner = dop.skb.make_learner(choose=study.ask())
    assert learner.fit_transform({}) == ("a", 13)
    assert params(learner) == {"data_op__0": 0, "data_op__1": 13}


@pytest.mark.parametrize(
    ("use_choose_from", "outcome_names"), [(False, None), (True, False), (True, True)]
)
def test_optuna_optimize_learner(use_choose_from, outcome_names):
    optuna = pytest.importorskip("optuna")

    if use_choose_from:
        if outcome_names:
            choice = skrub.choose_from(
                {"minus one": -1.0, "zero": 0.0, "two": 2.0, "four": 4.0}, name="x"
            )
        else:
            choice = skrub.choose_from((-1.0, 0.0, 2.0, 4.0), name="x")
        n_trials = 10
    else:
        choice = skrub.choose_float(-1.0, 7.0, name="x")
        n_trials = 50
    x = choice.as_data_op().skb.set_name("x_")
    err = (x - 2) ** 2

    def objective(trial):
        learner = err.skb.make_learner(choose=trial)
        return learner.fit_transform({})

    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    if use_choose_from:
        if outcome_names:
            assert study.best_params == {"0:x": "2:two"}
        else:
            assert study.best_params == {"0:x": "2:2.0"}
    else:
        assert list(study.best_params.keys()) == ["0:x"]
        assert study.best_params["0:x"] == pytest.approx(2.0, abs=0.01)

    # test both set_params(**best_params) or make_learner(choose=best_trial)
    learner_0 = err.skb.make_learner(choose=study.best_trial)
    learner_1 = err.skb.make_learner()
    learner_1.set_params(**study.best_params)
    for learner in [learner_0, learner_1]:
        assert learner.get_params()["data_op__0"] == pytest.approx(2.0, abs=0.01)
        truncated = learner.truncated_after("x_")
        assert truncated.fit_transform({}) == pytest.approx(2.0, abs=0.01)


def test_is_optuna_trial(monkeypatch):
    # verify we check if an object is an optuna trial without importing optuna.
    monkeypatch.delitem(sys.modules, "optuna", raising=False)
    from skrub._data_ops._skrub_namespace import _is_optuna_trial

    assert not _is_optuna_trial(0)
    assert "optuna" not in sys.modules


@pytest.mark.parametrize("how", ["deepcopy", "sklearn", "skb"])
def test_cloning_and_preview_data(how):
    e = skrub.var("a", 0) + skrub.var("b", 1)
    if how == "deepcopy":
        clone = copy.deepcopy(e)
    elif how == "sklearn":
        clone = e.__sklearn_clone__()
    else:
        clone = e.skb.clone()
    assert clone._skrub_impl.results == {}
    assert clone._skrub_impl.errors == {}
    assert clone._skrub_impl.metadata == {}
    if how == "skb":
        # This outputting "Evaluation of node..." in < Python 3.11
        msg = (
            "(No value has been provided for 'a')|(Evaluation of node <Var 'a'> failed)"
        )
        with pytest.raises(Exception, match=msg):
            clone.skb.eval()
    else:
        assert clone.skb.eval() == 1
    assert clone.skb.eval({"a": 10, "b": 2}) == 12


def test_data_op_impl():
    # misc bits to make codecove happy
    class A(_data_ops.DataOpImpl):
        _fields = ()

    a = _data_ops.DataOp(A())
    assert repr(a) == "<A>"
    # This is raising a RuntimeError in < Python 3.11
    with pytest.raises((NotImplementedError, RuntimeError)):
        a.skb.eval()


@pytest.mark.parametrize("why_no_wrap", ["numpy", "predictor", "how"])
@pytest.mark.parametrize("bad_param", ["cols", "how", "allow_reject"])
def test_apply_bad_params(why_no_wrap, bad_param):
    # When the estimator is a predictor or the input is a numpy array (not a
    # dataframe) (or how='no_wrap') the estimator can only be applied to the
    # full input without wrapping in ApplyToCols or ApplyToFrame. In this case
    # if the user passed a parameter that would require wrapping, such as
    # passing a value for `cols` that is not `all()`, or passing
    # how='cols' or allow_reject=True, we get an error.

    if why_no_wrap == bad_param == "how":
        return
    X_a, y_a = make_classification(random_state=0)
    X_df = pd.DataFrame(X_a, columns=[f"col_{i}" for i in range(X_a.shape[1])])

    X = skrub.X(X_a) if why_no_wrap == "numpy" else skrub.X(X_df)
    if why_no_wrap == "predictor":
        estimator = LogisticRegression()
        y = skrub.y(y_a)
    else:
        estimator = PassThrough()
        y = None
    how = "no_wrap" if why_no_wrap == "how" else "auto"
    # X is a numpy array: how must be no_wrap and selecting columns is not
    # allowed.
    if bad_param == "cols":
        if why_no_wrap == "numpy":
            cols = [0, 1]
        else:
            cols = ["col_0", "col_1"]
    else:
        cols = s.all()
    how = "cols" if bad_param == "how" else how
    allow_reject = True if bad_param == "allow_reject" else False

    with pytest.raises(
        (ValueError, RuntimeError),
        match=(
            r"(`cols` must be `all\(\)`|`how` must be 'auto'|`allow_reject` must be"
            r" False)"
        ),
    ):
        X.skb.apply(estimator, y=y, how=how, allow_reject=allow_reject, cols=cols)


def test_apply_invalid_how():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    X = skrub.var("X", df)
    t = PassThrough()
    for how in ["auto", "cols", "frame", "no_wrap"]:
        assert list(X.skb.apply(t, how=how).skb.eval().columns) == ["a", "b"]
    with pytest.raises(RuntimeError, match="`how` must be one of"):
        X.skb.apply(t, how="bad value")
    # TODO: remove when old names are dropped in 0.7.0
    with pytest.warns(FutureWarning, match="'columnwise' has been renamed to 'cols'"):
        wrapper = X.skb.apply(t, how="columnwise").skb.applied_estimator.skb.eval()
        assert isinstance(wrapper, ApplyToCols)


class Mul(BaseEstimator):
    def __init__(self, factor=1):
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return X * self.factor

    def transform(self, X):
        return X * self.factor


@pytest.mark.parametrize("use_choice", [False, True])
def test_apply_on_cols(use_choice):
    X_df = pd.DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})

    if use_choice:
        transformer = skrub.choose_from([Mul(10), Mul(100)], name="t")
    else:
        transformer = Mul(10)
    e = skrub.as_data_op(X_df).skb.apply(transformer, cols=["b"])
    out = e.skb.eval()
    expected = np.arange(6).reshape(2, -1).T * [1, 10]
    assert (out.values == expected).all()

    e = skrub.as_data_op(X_df).skb.apply(transformer, exclude_cols=["a"])
    out = e.skb.eval()
    assert (out.values == expected).all()


def test_apply_kwargs():
    class E(BaseEstimator):
        def fit(self, X, y=None, extra_f=None):
            assert extra_f == "kwarg for fit"
            return self

        def predict(self, X, extra_p=None):
            assert extra_p == "kwarg for predict"
            return 0

        def score(self, X, y=None, **kwargs):
            assert not kwargs
            return 0

    pred = skrub.var("a").skb.apply(
        E(),
        fit_kwargs={"extra_f": "kwarg for fit"},
        predict_kwargs={"extra_p": "kwarg for predict"},
    )
    learner = pred.skb.make_learner()
    learner.fit({"a": 0})
    learner.predict({"a": 0})
    learner.score({"a": 0})


def test_apply_transformer_kwargs():
    # check kwargs get passed correctly to the transformer for all 'how' values.
    class T(BaseEstimator):
        def fit_transform(self, X, y=None, extra_f=None):
            assert extra_f == "kwarg for fit_transform"
            return X

        def transform(self, X, extra_t=None):
            assert extra_t == "kwarg for transform"
            return X

    kwargs = {
        "fit_transform_kwargs": {"extra_f": "kwarg for fit_transform"},
        "transform_kwargs": {"extra_t": "kwarg for transform"},
    }
    learner = (
        skrub.var("df")
        .skb.apply(T(), how="no_wrap", **kwargs)
        .skb.apply(T(), how="cols", **kwargs)
        .skb.apply(T(), how="frame", **kwargs)
        .skb.make_learner()
    )
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    learner.fit({"df": df})
    learner.fit_transform({"df": df})
    learner.transform({"df": df})


def test_apply_kwargs_evaluation():
    class E(BaseEstimator):
        def fit(self, X, y=None, extra_f=None):
            assert extra_f == "kwarg for fit"
            return self

        def predict(self, X, extra_p=None):
            assert extra_p == "kwarg for predict"
            return 0

        def predict_proba(self, X, **kwargs):
            assert not kwargs
            return 0

        def score(self, X, y=None, **kwargs):
            assert not kwargs
            return 0

    get_extra_f_n_calls = 0

    @skrub.deferred
    def get_extra_f(a):
        nonlocal get_extra_f_n_calls
        get_extra_f_n_calls += 1
        return "kwarg for fit"

    get_predict_kwargs_n_calls = 0

    @skrub.deferred
    def get_predict_kwargs(a):
        nonlocal get_predict_kwargs_n_calls
        get_predict_kwargs_n_calls += 1
        return {"extra_p": "kwarg for predict"}

    get_predict_proba_kwargs_n_calls = 0

    @skrub.deferred
    def get_predict_proba_kwargs(a):
        nonlocal get_predict_proba_kwargs_n_calls
        get_predict_proba_kwargs_n_calls += 1
        # None is a valid value for kwargs, get translated to {}
        return None

    assert get_extra_f_n_calls == 0
    a = skrub.var("a")
    pred = a.skb.apply(
        E(),
        fit_kwargs={"extra_f": get_extra_f(a)},
        predict_kwargs=get_predict_kwargs(a),
        predict_proba_kwargs=get_predict_proba_kwargs(a),
    )
    learner = pred.skb.make_learner()
    assert get_extra_f_n_calls == 0
    assert get_predict_kwargs_n_calls == 0
    assert get_predict_proba_kwargs_n_calls == 0

    learner.fit({"a": 0})
    assert get_extra_f_n_calls == 1
    # the kwargs for predict have not been evaluated when calling fit()
    assert get_predict_kwargs_n_calls == 0
    assert get_predict_proba_kwargs_n_calls == 0

    learner.predict({"a": 0})
    assert get_extra_f_n_calls == 1
    assert get_predict_kwargs_n_calls == 1
    assert get_predict_proba_kwargs_n_calls == 0

    learner.predict_proba({"a": 0})
    assert get_extra_f_n_calls == 1
    assert get_predict_kwargs_n_calls == 1
    assert get_predict_proba_kwargs_n_calls == 1

    learner.score({"a": 0})
    assert get_extra_f_n_calls == 1
    assert get_predict_kwargs_n_calls == 1
    assert get_predict_proba_kwargs_n_calls == 1


def test_apply_bad_kwargs():
    with pytest.raises(
        (TypeError, RuntimeError),
        match=(
            r".*The `fit_kwargs` passed to `\.skb\.apply\(\)` should be a dict of named"
            r" arguments"
        ),
    ):
        skrub.X(0).skb.apply(DummyRegressor(), fit_kwargs="BAD", y=skrub.y(0))


def test_concat_horizontal_duplicate_cols():
    X_df = pd.DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
    X = skrub.X()
    e = X.skb.concat([X], axis=1)
    learner = e.skb.make_learner()
    out_1 = learner.fit_transform({"X": X_df})
    out_2 = learner.transform({"X": X_df})
    assert len(set(out_1.columns)) == len(out_1.columns) == 4
    assert list(out_1.columns) == list(out_2.columns)


def test_concat_vertical_duplicate_cols():
    X_df1 = pd.DataFrame({"a": [0, 1], "b": [2, 3]})
    X_df2 = pd.DataFrame({"a": [4, 5], "b": [6, 7]})  # Same columns
    X1 = skrub.var("X1", pd.DataFrame({"a": [0, 1], "b": [2, 3]}))
    X2 = skrub.var("X2", pd.DataFrame({"a": [4, 5], "b": [6, 7]}))

    e = X1.skb.concat([X2], axis=0)
    assert isinstance(e, skrub.DataOp)

    learner = e.skb.make_learner()
    data_dict = {"X1": X_df1, "X2": X_df2}
    out_1 = learner.fit_transform(data_dict)
    out_2 = learner.transform(data_dict)

    assert out_1.shape[1] == out_2.shape[1] == 2
    assert out_1.shape[0] == out_2.shape[0] == 4


def test_concat_non_str_colname():
    int_columns = pd.DataFrame({0: [1, 2], 1: [3, 4]})
    string_columns = pd.DataFrame({"0": [1, 2], "1": [3, 4]})

    # check that a warning is raised because of non-string column name
    with pytest.warns(
        UserWarning, match="Some dataframe column names are not strings:"
    ):
        skrub.as_data_op(int_columns).skb.concat(
            [skrub.as_data_op(string_columns)], axis=1
        )
    with pytest.warns(
        UserWarning, match="Some dataframe column names are not strings:"
    ):
        skrub.as_data_op(int_columns).skb.concat(
            [skrub.as_data_op(int_columns)], axis=1
        )

    # no warnings raised when all column names are strings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        skrub.as_data_op(string_columns).skb.concat(
            [skrub.as_data_op(string_columns)], axis=1
        )

    # check that no warning is raised when concatenating vertically
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        skrub.as_data_op(int_columns).skb.concat(
            [skrub.as_data_op(int_columns)], axis=0
        )


def test_get_vars():
    a = skrub.var("a")
    b = skrub.var("b")
    c = (a + b).skb.set_name("c")
    d = c + c
    assert list(d.skb.get_vars().keys()) == ["a", "b"]
    assert d.skb.get_vars()["a"] is a
    assert list(d.skb.get_vars(all_named_ops=True).keys()) == ["a", "b", "c"]


@pytest.mark.parametrize("needs_data", [False, True])
@pytest.mark.parametrize("has_preview", [False, True])
@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("with_scoring", [False, True])
def test_estimator_is_a_data_op(needs_data, has_preview, regression, with_scoring):
    # Check that the data_op, learner and search estimators behave well when
    # the estimator passed to apply is a data op
    if regression:
        X_a, y_a = make_regression(random_state=0)
    else:
        X_a, y_a = make_classification(random_state=0)
    X_df = pd.DataFrame(X_a).rename(columns=str)
    if has_preview:
        X, y = skrub.X(X_df), skrub.y(y_a)
    else:
        X, y = skrub.X(), skrub.y()
    if needs_data:
        # In this case the estimator's automated preview cannot be computed
        # because it needs a value from the environment, the value is not known
        # until we fit the learner.

        def get_vectorizer(X):
            return skrub.TableVectorizer()

        vectorizer = X.skb.apply_func(get_vectorizer)

        def get_predictor(X):
            return Ridge() if regression else LogisticRegression()

        predictor = X.skb.apply_func(get_predictor)
    else:
        # In this case the estimator can be evaluated in the automated preview
        # when the data op is created.
        vectorizer = skrub.as_data_op(skrub.TableVectorizer())
        predictor = skrub.as_data_op(Ridge() if regression else LogisticRegression())
    pred = X.skb.apply(vectorizer).skb.apply(predictor, y=y)
    # no information about the estimator: we expose all methods and default to
    # 'transformer' estimator type.
    learner = pred.skb.make_learner()
    assert learner.__skrub_to_Xy_pipeline__({})._estimator_type == "transformer"
    assert hasattr(learner, "predict")
    assert hasattr(pred.skb.make_randomized_search(), "predict")
    if has_preview:
        assert pred.skb.preview().shape == y_a.shape
    env = {"X": X_df, "y": y_a}
    assert pred.skb.eval(env).shape == y_a.shape
    search = pred.skb.make_grid_search(cv=2).fit(env)
    min_score = 0.3 if regression else 0.7
    assert search.best_score_ > min_score
    if with_scoring:
        scoring = "r2" if regression else "accuracy"
    else:
        scoring = None
    res = skrub.cross_validate(
        pred.skb.make_grid_search(cv=2, scoring=scoring),
        environment=env,
        cv=2,
        scoring=scoring,
    )
    assert res["test_score"].mean() > min_score


def test_apply_no_sklearn_tags():
    # applying an estimator that does not define __sklearn_tags__
    class Twice:
        def fit(self, X, y=None):
            return self

        def fit_transform(self, X, y=None):
            return X * 2

        def transform(self, X):
            return X * 2

        def get_params(self, deep=True):
            return {}

        def set_params(self):
            return self

    learner = skrub.var("a").skb.apply(Twice()).skb.make_learner()
    assert learner.fit_transform({"a": 1}) == 2
    xy_learner = learner.__skrub_to_Xy_pipeline__({})
    assert xy_learner._estimator_type == "transformer"
    if hasattr(xy_learner, "__sklearn_tags__"):
        # Old scikit-learn versiond don't have __sklearn_tags__
        assert xy_learner.__sklearn_tags__().estimator_type is None


def test_class_skb():
    from skrub._data_ops._skrub_namespace import SkrubNamespace

    assert skrub.DataOp.skb is SkrubNamespace


def test_copy_attrs():
    # non-regression for #1781 some attributes could be missing after
    # .set_name(), .mark_as_X() etc.
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    out = (
        skrub.var("X", df)
        .skb.apply(PassThrough())
        .skb.mark_as_X()
        .skb.set_name("transform")
    )
    assert isinstance(out.skb.applied_estimator.skb.eval().transformer_, PassThrough)
