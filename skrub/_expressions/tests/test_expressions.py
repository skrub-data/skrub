import pytest
from sklearn.linear_model import LogisticRegression

import skrub


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
