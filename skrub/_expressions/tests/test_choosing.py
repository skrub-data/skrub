import numpy as np
import pytest

import skrub
from skrub._expressions import _choosing


@pytest.mark.parametrize("name", ["the_param", "my_description__the_param"])
def test_choice_unpacking(name):
    for choice in [
        skrub.choose_from([10, 20], name=name),
        skrub.choose_from({"a": 10, "b": 20, name: 30}, name=name),
        skrub.choose_float(0.1, 10.0, log=True, name=name),
        skrub.choose_float(0.1, 10.0, log=True, n_steps=3, name=name),
        skrub.choose_int(5, 15, n_steps=5, name=name),
        skrub.choose_bool(name=name),
        skrub.optional("value", name=name),
    ]:
        assert {**choice} == {"the_param": choice}
        with pytest.raises(KeyError):
            choice["a"]
    with pytest.raises(TypeError):
        skrub.choose_from([10, 20], name=name)[0]


@pytest.mark.parametrize("name", [None, 0, skrub.var("name", "the_name")])
def test_bad_name(name):
    with pytest.raises(TypeError, match=".*must be a `str`"):
        skrub.choose_from({"a": 0}, name=name)
    with pytest.raises(TypeError, match=".*must be a `str`"):
        skrub.choose_int(0, 10, name=name)
    with pytest.raises(TypeError, match=".*must be a `str`"):
        skrub.choose_int(0, 10, n_steps=5, name=name)


def test_discretized_choice_iter():
    c = skrub.choose_int(10, 20, n_steps=6, name="c")
    assert list(c) == list(range(10, 21, 2))
    assert len(c) == 6
    assert c[1] == 12
    c = skrub.choose_int(10, 20, n_steps=1, name="c")
    assert list(c) == [10]
    assert len(c) == 1
    assert c[0] == 10
    c = skrub.choose_int(10, 20, n_steps=100, name="c")
    assert list(c) == list(range(10, 21))
    assert len(c) == 11
    assert c[1] == 11
    c = skrub.choose_float(10, 20, n_steps=100, name="c")
    assert (list(c) == np.linspace(10, 20, 100)).all()
    assert len(c) == 100
    with pytest.raises(ValueError, match=".*>= 1"):
        skrub.choose_float(10.0, 20.0, n_steps=0, name="c")


def test_bad_outcomes():
    with pytest.raises(ValueError, match="Choice should be given at least one outcome"):
        skrub.choose_from([], name="c")
    with pytest.raises(ValueError, match="Choice should be given at least one outcome"):
        skrub.choose_from({}, name="c")
    with pytest.raises(TypeError, match="Outcome names.*should be of type `str`"):
        skrub.choose_from({0: "a", 1: "b"}, name="c")
    # the rest is not feasible through the public functions choose_from etc. but
    # Choice objects check those invariants anyway so we trigger them here for
    # coverage.
    with pytest.raises(ValueError):
        _choosing.Choice([10, 20], outcome_names=["a"], name="c")
    with pytest.raises(ValueError):
        _choosing.Choice([10, 20], outcome_names=["a", "a"], name="c")


def test_bad_bounds():
    with pytest.raises(ValueError, match="'high' must be greater"):
        skrub.choose_int(20, 10, name="c")
    with pytest.raises(ValueError, match=".*'low' must be > 0"):
        skrub.choose_float(0.0, 10.0, log=True, name="c")


def test_as_expr():
    c = skrub.choose_from([10, 20, 30], name="c")
    m = c.match({10: "ten", 20: "twenty"}, default="?")
    assert c.as_expr().skb.eval() == 10
    assert c.as_expr().skb.eval({"c": 20}) == 20
    assert m.as_expr().skb.eval() == "ten"
    assert m.as_expr().skb.eval({"c": 30}) == "?"
    with pytest.raises(KeyError):
        assert m.as_expr().skb.eval({"c": 40})


def test_get_chosen_or_default():
    c = skrub.choose_from([10, 20], name="c")
    assert _choosing.get_chosen_or_default(c) == 10
    assert _choosing.get_default(c) == 10
    c.chosen_outcome_idx = 1
    assert _choosing.get_chosen_or_default(c) == 20
    assert _choosing.get_default(c) == 10

    c = skrub.choose_from([10, 20], name="c")
    m = c.match({10: "ten", 20: "twenty"})
    assert _choosing.get_chosen_or_default(m) == "ten"
    assert _choosing.get_default(m) == "ten"
    c.chosen_outcome_idx = 1
    assert _choosing.get_chosen_or_default(m) == "twenty"
    assert _choosing.get_default(m) == "ten"

    c = skrub.choose_float(10.0, 20.0, name="c")
    assert _choosing.get_chosen_or_default(c) == 15.0
    assert _choosing.get_default(c) == 15.0
    c.chosen_outcome = 12.0
    assert _choosing.get_chosen_or_default(c) == 12.0
    assert _choosing.get_default(c) == 15.0


def test_match():
    assert not hasattr(skrub.choose_int(0, 10, n_steps=6, name="N"), "match")
    c = skrub.choose_from(["a", "b", "c"], name="c")

    # m and mm without default
    m = c.match({"a": 10, "b": 20, "c": 30})
    assert m.outcome_mapping == {"a": 10, "b": 20, "c": 30}
    mm = m.match({10: "ten", 20: "twenty", 30: "thirty"})
    assert mm.choice is c
    assert mm.outcome_mapping == {"a": "ten", "b": "twenty", "c": "thirty"}

    # m without default, mm with default
    m = c.match({"a": 10, "b": 20, "c": 30})
    assert m.outcome_mapping == {"a": 10, "b": 20, "c": 30}
    mm = m.match({10: "ten", 20: "twenty"}, default="?")
    assert mm.choice is c
    assert mm.outcome_mapping == {"a": "ten", "b": "twenty", "c": "?"}

    # mm without default, m with default
    m = c.match({"a": 10, "b": 20}, default="?")
    assert m.outcome_mapping == {"a": 10, "b": 20, "c": "?"}
    mm = m.match({10: "ten", 20: "twenty", "?": "??"})
    assert mm.choice is c
    assert mm.outcome_mapping == {"a": "ten", "b": "twenty", "c": "??"}

    # both with default
    m = c.match({"a": 10, "b": 20}, default="?")
    assert m.outcome_mapping == {"a": 10, "b": 20, "c": "?"}
    mm = m.match({10: "ten", 20: "twenty"}, default="??")
    assert mm.choice is c
    assert mm.outcome_mapping == {"a": "ten", "b": "twenty", "c": "??"}


@pytest.mark.parametrize("test_class", ["choice", "match"])
def test_bad_match_mappings(test_class):
    c = skrub.choose_from(["a", "b", "c"], name="c")
    if test_class == "match":
        c = c.match({"a": "a", "b": "b", "c": "c"})
    with pytest.raises(
        ValueError,
        match="The following outcomes do not have a corresponding key "
        r"in the mapping provided to `match\(\)`: \{'c'\}",
    ):
        c.match({"a": 10, "b": 20})
    with pytest.raises(
        ValueError,
        match="The following keys were found in the mapping provided "
        r"to `match\(\)` but are not possible choice outcomes: \{'d'\}",
    ):
        c.match({"a": 10, "b": 20, "c": 30, "d": 40})
    c = skrub.choose_from([skrub.X(), skrub.X() + 10], name="c")
    with pytest.raises(
        TypeError,
        match=r"To use `match\(\)`, all choice outcomes must be hashable. "
        "unhashable type: 'Expr'",
    ):
        c.match({1: "one", 11: "eleven"})
