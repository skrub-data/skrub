import copy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression

import skrub
from skrub import selectors as s
from skrub._expressions import _expressions
from skrub._utils import PassThrough


def test_simple_expression():
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
    e = skrub.as_expr(LogisticRegression(C=c))
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
    with pytest.raises(KeyError):
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


def test_predictor_as_df_transformer():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    pred = skrub.X().skb.apply(DummyRegressor(), y=skrub.y())
    estimator = pred.skb.get_estimator()
    expected = pd.DataFrame({"a": [2.0, 2.0, 2.0], "b": [20.0, 20.0, 20.0]})
    assert_frame_equal(estimator.fit_transform({"X": X, "y": X}), expected)
    assert_frame_equal(estimator.transform({"X": X, "y": X}), expected)


def test_get_estimator():
    e = (skrub.var("a", 0) + skrub.var("b", 1)).skb.get_estimator()
    assert e.fit_transform({"a": 10, "b": 2}) == 12
    assert e.transform({"a": 100, "b": 30}) == 130


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
    if how == "skb":
        with pytest.raises(Exception, match="No value value has been provided for 'a'"):
            clone.skb.eval()
    else:
        assert clone.skb.eval() == 1
    assert clone.skb.eval({"a": 10, "b": 2}) == 12


def test_expr_impl():
    # misc bits to make codecove happy
    class A(_expressions.ExprImpl):
        _fields = ()

    a = _expressions.Expr(A())
    assert repr(a) == "<A>"
    with pytest.raises(NotImplementedError):
        a.skb.eval()


@pytest.mark.parametrize("why_full_frame", ["numpy", "predictor", "how"])
@pytest.mark.parametrize("bad_param", ["cols", "how", "allow_reject"])
def test_apply_bad_params(why_full_frame, bad_param):
    # When the estimator is a predictor or the input is a numpy array (not a
    # dataframe) (or how='full_frame') the estimator can only be applied to the
    # full input without wrapping in OnEachColumn or OnSubFrame. In this case
    # if the user passed a parameter that would require wrapping, such as
    # passing a value for `cols` that is not `all()`, or passing
    # how='columnwise' or allow_reject=True, we get an error.

    if why_full_frame == bad_param == "how":
        return
    X_a, y_a = make_classification(random_state=0)
    X_df = pd.DataFrame(X_a, columns=[f"col_{i}" for i in range(X_a.shape[1])])

    X = skrub.X(X_a) if why_full_frame == "numpy" else skrub.X(X_df)
    if why_full_frame == "predictor":
        estimator = LogisticRegression()
        y = skrub.y(y_a)
    else:
        estimator = PassThrough()
        y = None
    how = "full_frame" if why_full_frame == "how" else "auto"
    # X is a numpy array: how must be full_frame and selecting columns is not
    # allowed.
    if bad_param == "cols":
        if why_full_frame == "numpy":
            cols = [0, 1]
        else:
            cols = ["col_0", "col_1"]
    else:
        cols = s.all()
    how = "columnwise" if bad_param == "how" else how
    allow_reject = True if bad_param == "allow_reject" else False

    with pytest.raises((ValueError, RuntimeError), match=""):
        X.skb.apply(estimator, y=y, how=how, allow_reject=allow_reject, cols=cols)


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
    e = skrub.as_expr(X_df).skb.apply(transformer, cols=["b"])
    out = e.skb.eval()
    expected = np.arange(6).reshape(2, -1).T * [1, 10]
    assert (out.values == expected).all()

    e = skrub.as_expr(X_df).skb.apply(transformer, exclude_cols=["a"])
    out = e.skb.eval()
    assert (out.values == expected).all()


def test_concat_horizontal_duplicate_cols():
    X_df = pd.DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
    X = skrub.X()
    e = X.skb.concat_horizontal([X])
    estimator = e.skb.get_estimator()
    out_1 = estimator.fit_transform({"X": X_df})
    out_2 = estimator.transform({"X": X_df})
    assert len(set(out_1.columns)) == len(out_1.columns) == 4
    assert list(out_1.columns) == list(out_2.columns)


def test_class_skb():
    from skrub._expressions._skrub_namespace import SkrubNamespace

    assert skrub.Expr.skb is SkrubNamespace
