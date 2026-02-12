import time
from collections import namedtuple

import pytest

import skrub
from skrub._data_ops import _evaluation


def test_caching():
    a = skrub.var("a", 100)
    b = a + a
    c = b + a
    d = c + a

    def check_cache_during_fit_transform():
        # we are running in "preview_fit_transform" mode so only the "fit_transform"
        # items in the cache should be touched, not the "preview".

        # before the first node is evaluated all caches are empty
        assert a._skrub_impl.results == {"preview": 100}
        assert b._skrub_impl.results == {"preview": 200}
        assert c._skrub_impl.results == {"preview": 300}
        assert d._skrub_impl.results == {"preview": 400}
        yield
        # a has been computed
        assert a._skrub_impl.results == {"preview": 100, "fit_transform": 10}
        assert b._skrub_impl.results == {"preview": 200}
        assert c._skrub_impl.results == {"preview": 300}
        assert d._skrub_impl.results == {"preview": 400}
        yield
        # b has been computed, a is still needed so both are in the cache
        assert a._skrub_impl.results == {"preview": 100, "fit_transform": 10}
        assert b._skrub_impl.results == {"preview": 200, "fit_transform": 20}
        assert c._skrub_impl.results == {"preview": 300}
        assert d._skrub_impl.results == {"preview": 400}
        yield
        # c has been computed, b is not needed any more, a is still needed for d
        assert a._skrub_impl.results == {"preview": 100, "fit_transform": 10}
        assert b._skrub_impl.results == {"preview": 200}
        assert c._skrub_impl.results == {"preview": 300, "fit_transform": 30}
        assert d._skrub_impl.results == {"preview": 400}
        yield
        # d has been computed, a and c are not needed anymore
        assert a._skrub_impl.results == {"preview": 100}
        assert b._skrub_impl.results == {"preview": 200}
        assert c._skrub_impl.results == {"preview": 300}
        assert d._skrub_impl.results == {"preview": 400, "fit_transform": 40}
        yield

    # the preview cache has been filled eagerly when defining the DataOp
    assert a._skrub_impl.results == {"preview": 100}
    assert b._skrub_impl.results == {"preview": 200}
    assert c._skrub_impl.results == {"preview": 300}
    assert d._skrub_impl.results == {"preview": 400}

    check = check_cache_during_fit_transform()
    next(check)
    _evaluation.evaluate(
        d,
        mode="fit_transform",
        environment={"a": 10},
        clear=True,
        callbacks=((lambda e, r: next(check)),),
    )

    # the check generator is exhausted (we reached the last yield)
    assert next(check, "finished") == "finished"

    # and the last remaining result has been cleared from the cache as well
    assert a._skrub_impl.results == {"preview": 100}
    assert b._skrub_impl.results == {"preview": 200}
    assert c._skrub_impl.results == {"preview": 300}
    assert d._skrub_impl.results == {"preview": 400}

    _evaluation.clear_results(d)

    # after clearing all results
    assert a._skrub_impl.results == {}
    assert b._skrub_impl.results == {}
    assert c._skrub_impl.results == {}
    assert d._skrub_impl.results == {}


def test_caching_in_special_data_ops():
    # DataOp that need to skip evaluation of some branches based on a
    # condition like if_else and match are somewhat special cases so we check
    # here that their cache gets populated correctly as well.
    a = skrub.var("a")
    b = skrub.var("b")
    c = skrub.var("c")
    d = a.skb.if_else(b, c)
    e = d.skb.match({"B": "BE"}, default="CE")
    _evaluation.evaluate(e, mode="fit_transform", environment={"a": True, "b": "B"})
    assert a._skrub_impl.results == {"fit_transform": True}
    assert b._skrub_impl.results == {"fit_transform": "B"}
    assert c._skrub_impl.results == {}
    assert d._skrub_impl.results == {"fit_transform": "B"}
    assert e._skrub_impl.results == {"fit_transform": "BE"}


def test_needs_eval():
    # needs_eval() is used to check if a collection contains some skrub
    # DataOp or choice. problems with cyclical references are handled
    # separately, so when it finds one needs_eval must just return False.
    globals_ = {}
    globals_["globals_"] = globals_
    assert not _evaluation.needs_eval(globals_)

    assert _evaluation.needs_eval(
        {
            "a": [
                0,
                skrub.TableVectorizer(
                    **skrub.choose_int(10, 20, name="cardinality_threshold")
                ),
            ]
        }
    )


def test_find_node_by_name():
    a = skrub.var("a")
    X = skrub.X()
    b = (X + a).skb.set_name("b")
    c = skrub.choose_from([1, 2], name="c")
    d = b + c
    e = d + d
    assert _evaluation.find_node_by_name(e, "a") is a
    assert _evaluation.find_node_by_name(e, "b") is b
    assert _evaluation.find_node_by_name(e, "X") is X
    assert _evaluation.find_node_by_name(e, "c") is c
    assert _evaluation.find_node_by_name(e, "d") is None


#
# cloning
#


def test_clone_preserves_structure():
    a = skrub.var("a")
    c = skrub.choose_from([1, 2], name="c")
    e = skrub.as_data_op([c, c, a, a])
    clone = e.skb.clone()
    assert clone.skb.describe_steps() == e.skb.describe_steps()
    assert clone.skb.describe_param_grid() == e.skb.describe_param_grid()
    assert _evaluation.param_grid(clone) == _evaluation.param_grid(e)


#
# param grid
#


def test_empty_param_grid():
    """
    >>> import skrub
    >>> print(skrub.X().skb.describe_param_grid())
    <empty parameter grid>
    """
    assert _evaluation.param_grid(None) == [{}]


def test_param_grid_nested_choices():
    c0 = skrub.choose_from([10, 20, 30], name="c0")
    c1 = skrub.choose_from([11, 21, 22, 24], name="c1")
    c2 = skrub.choose_from([{"C": c0}, {"C": c1}], name="c2")
    c3 = skrub.choose_from([12, 22, 40, 50, 60], name="c3")
    e = skrub.as_data_op([c2, c3])
    assert (
        e.skb.describe_param_grid()
        == """\
- c3: [12, 22, 40, 50, 60]
  c2: {'C': choose_from([10, 20, 30], name='c0')}
  c0: [10, 20, 30]
- c3: [12, 22, 40, 50, 60]
  c2: {'C': choose_from([11, 21, 22, 24], name='c1')}
  c1: [11, 21, 22, 24]
"""
    )
    assert _evaluation.param_grid(e) == [
        {2: [0], 0: [0, 1, 2], 3: [0, 1, 2, 3, 4]},
        {2: [1], 1: [0, 1, 2, 3], 3: [0, 1, 2, 3, 4]},
    ]


def test_param_grid_choice_before_X():
    c0 = skrub.choose_from([10, 20], name="c0")
    c1 = skrub.choose_float(0.0, 1.0, name="c1")
    b = skrub.var("a") + c0 + c1
    c2 = skrub.choose_from([12, 22], name="c2")
    c = b + c2
    assert _evaluation.param_grid(c) == [
        {0: [0, 1], 1: skrub.choose_float(0.0, 1.0, name="c1"), 2: [0, 1]}
    ]
    assert (
        c.skb.describe_param_grid()
        == """\
- c0: [10, 20]
  c1: choose_float(0.0, 1.0, name='c1')
  c2: [12, 22]
"""
    )

    with pytest.warns(
        UserWarning,
        match=(
            r"The following choices are used in the construction of "
            r"X or y.*\[choose_from\(\[10, 20\], name='c0'\), "
            r"choose_float\(0.0, 1.0, name='c1'\)\]"
        ),
    ):
        c0 = skrub.choose_from([10, 20], name="c0")
        c1 = skrub.choose_float(0.0, 1.0, name="c1")
        b = (skrub.var("a") + c0 + c1).skb.mark_as_X()
        c2 = skrub.choose_from([12, 22], name="c2")
        c = b + c2
        # the choices that are before X are clamped to their default value (0
        # for choice 0 and 0.5 for choice 1)
        assert _evaluation.param_grid(c) == [
            {
                0: [0],
                1: [0.5],
                2: [0, 1],
            }
        ]
        assert (
            c.skb.describe_param_grid().replace("np.float64(0.5)", "0.5")
            == """\
- c0: 10
  c1: [0.5]
  c2: [12, 22]
"""
        )


def test_unnamed_choices():
    """
    >>> import skrub

    >>> a = skrub.choose_bool()
    >>> b = skrub.choose_bool()
    >>> c = skrub.choose_bool()
    >>> d = skrub.choose_from({'a': a, 'b': b, 'c': c})
    >>> x = skrub.as_data_op([a, b, c, d])
    >>> print(x.skb.describe_param_grid())
    - choose_bool(): [True, False]
      choose_bool()_1: [True, False]
      choose_bool()_2: [True, False]
      choose_from({'a': …, 'b': …, 'c': …}): 'a'
    - choose_bool(): [True, False]
      choose_bool()_1: [True, False]
      choose_bool()_2: [True, False]
      choose_from({'a': …, 'b': …, 'c': …}): 'b'
    - choose_bool(): [True, False]
      choose_bool()_1: [True, False]
      choose_bool()_2: [True, False]
      choose_from({'a': …, 'b': …, 'c': …}): 'c'
    """
    a = skrub.choose_int(1, 5)
    b = skrub.choose_int(1, 5)
    c = skrub.choose_int(1, 5, name="c")
    e = a.as_data_op() + b + c
    assert e.skb.eval() == 9
    assert e.skb.eval({"c": 5}) == 11
    assert _evaluation.param_grid(e) == [{0: a, 1: b, 2: c}]


#
# misc details mostly for code coverage
#


def test_clone_bad_sklearn_protocol():
    class A:
        __sklearn_clone__ = 0

    assert isinstance(_evaluation.clone(A()), A)


def test_describe_steps():
    """
    >>> import skrub


    >>> @skrub.deferred
    ... def func(x, y):
    ...     return x + y


    >>> a = skrub.var("a")
    >>> b = a + a
    >>> c = (
    ...     func(a, skrub.var("b"))
    ...     .skb.apply(skrub.TableVectorizer())
    ...     .amethod(skrub.as_data_op(10))
    ...     .skb.concat([b], axis=1)
    ...     + skrub.choose_bool(name="?").as_data_op()
    ...     + skrub.X().skb.if_else(3, b)[skrub.var("item")].b
    ... )
    >>> print(c.skb.describe_steps())
    Var 'a'
    Var 'b'
    Call 'func'
    Apply TableVectorizer
    Value int
    CallMethod 'amethod'
    ( Var 'a' )*
    ( Var 'a' )*
    BinOp: add
    Concat: 2 dataframes
    Value BoolChoice
    BinOp: add
    Var 'X'
    ( Var 'a' )*
    ( Var 'a' )*
    ( BinOp: add )*
    IfElse <Var 'X'> ? 3 : <BinOp: add>
    Var 'item'
    GetItem <Var 'item'>
    GetAttr 'b'
    BinOp: add
    * Cached, not recomputed
    """


def _generator_result(g):
    while True:
        try:
            next(g)
        except StopIteration as e:
            return e.value


def test_as_gen():
    def f():
        return 0

    assert _generator_result(_evaluation._as_gen(f)()) == 0

    def g():
        yield 0
        return 1

    assert _evaluation._as_gen(g) is g


def test_eval_duration():
    def after(obj, duration):
        time.sleep(duration)
        return obj

    a = skrub.as_data_op(1)
    b = skrub.as_data_op(2).skb.apply_func(after, 3.0)
    c = a + b

    def get_duration(dop):
        return dop._skrub_impl.metadata["preview"]["eval_duration"]

    # timing has large variance in CI, prob. due to not having 100% of CPU
    assert get_duration(a) == pytest.approx(0.0, abs=0.5)
    assert get_duration(b) == pytest.approx(3.0, abs=0.5)
    assert get_duration(c) == pytest.approx(0.0, abs=0.5)


def test_eval_builtin_sequence_subclass():
    # eval() recurses into some built-in collections. When evaluating a list,
    # we evaluate each of the items then construct a new list with the items'
    # values. We should not do it for the collection's subclasses because their
    # initialization may work differently, we do not know enough to safely
    # reconstruct the collection of values. For example tuple subclasses
    # created with namedtuple cannot be initialized with a sequence.
    # subclasses.

    Pair = namedtuple("Pair", ["first", "second"])
    p = Pair(1, 2)
    assert skrub.as_data_op(p).skb.eval() is p
    # this means items only get evaluated if they are stored in a builtin
    # collection
    p = Pair(skrub.as_data_op(1) + skrub.as_data_op(2), skrub.as_data_op(3))
    first = skrub.as_data_op(p).skb.eval()[0]
    assert isinstance(first, skrub.DataOp)
    first = skrub.as_data_op(tuple(p)).skb.eval()[0]
    assert isinstance(first, int)
    assert first == 3

    # same for dicts
    class Dict(dict):
        pass

    d = Dict()
    assert skrub.as_data_op(d).skb.eval() is d


def test_eval_fit():
    # evaluate() does no special handling of "fit" mode.
    # the learner's fit() discards the result and returns self.
    dop = skrub.as_data_op(1)
    assert _evaluation.evaluate(dop, mode="fit") == 1
    learner = dop.skb.make_learner()
    assert learner.fit({}) is learner
