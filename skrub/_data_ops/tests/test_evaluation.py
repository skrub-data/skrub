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
        callbacks=((lambda e, r, **kwargs: next(check)),),
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
    # note describe_steps() shows which nodes are reused, so comparing its
    # output for the original and the clone checks the graph's structure, not
    # only the sequence of operations.
    assert clone.skb.describe_steps() == e.skb.describe_steps()
    assert clone.skb.describe_param_grid() == e.skb.describe_param_grid()
    assert _evaluation.param_grid(clone) == _evaluation.param_grid(e)


def test_clone_structure_and_replace():
    a = skrub.var("a", 1)
    b = skrub.var("b", 2)
    shared = a + b
    data_op = shared * shared + a

    clone = _evaluation.clone(data_op)
    graph = _evaluation.graph(clone)
    # the clone is a DAG like the original, it has not been turned into a tree
    assert len(graph["nodes"]) == len(_evaluation.nodes(data_op)) == 5
    # 'a' is used by the addition and by the multiplication's parent
    assert len(graph["parents"][0]) == 2
    assert clone.skb.eval() == data_op.skb.eval() == 10

    # nodes for which the caller provides a replacement are substituted and
    # their children are not cloned.
    replacement = skrub.var("z", 99)
    replaced = _evaluation.clone(data_op, replace={id(shared): replacement})
    nodes = _evaluation.nodes(replaced)
    assert len(nodes) == 4
    # 'b' was only reachable through the node that has been replaced
    assert {n._skrub_impl.name for n in nodes} == {"z", "a", None}
    assert replaced.skb.eval() == 99 * 99 + 1


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


def test_param_grid_shared_node_choices():
    # A node that contains a choice can be reached through several paths. The
    # choices it contains must be listed in the sub-grid of each of the choice
    # outcomes from which it can be reached, so they remain tunable in all of
    # them. This is why _ChoiceGraph tracks the nodes it has already visited
    # separately for each outcome rather than only once for the whole
    # traversal.
    a = skrub.var("a", 1)
    inner = a * skrub.choose_int(1, 3, name="inner")
    outer = skrub.choose_from({"x": inner, "y": inner + 1}, name="outer")
    data_op = outer.as_data_op() + inner
    choice_names = _evaluation.choice_graph(data_op)["choice_display_names"]
    grid = _evaluation.param_grid(data_op)
    assert len(grid) == 2
    for sub_grid in grid:
        assert "inner" in {str(choice_names[c]) for c in sub_grid}
    assert (
        data_op.skb.describe_param_grid()
        == """\
- inner: choose_int(1, 3, name='inner')
  outer: 'x'
- inner: choose_int(1, 3, name='inner')
  outer: 'y'
"""
    )


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
    Var 'a' -> _0
    Var 'b'
    Call 'func'
    Apply TableVectorizer
    Value int
    CallMethod 'amethod'
    Load _0 (Var 'a')
    Load _0 (Var 'a')
    BinOp: add -> _6
    Concat: 2 tables
    Value BoolChoice
    BinOp: add
    Var 'X'
    Load _6 (BinOp: add)
    IfElse <Var 'X'> ? 3 : <BinOp: add>
    Var 'item'
    GetItem <Var 'item'>
    GetAttr 'b'
    BinOp: add
    """


def test_describe_steps_labels_only_reused_nodes():
    """
    >>> import skrub
    >>> a = skrub.var("a", 1)
    >>> b = skrub.var("b", 2)
    >>> c = a + b

    when nothing is reused no node is numbered

    >>> print(c.skb.describe_steps())
    Var 'a'
    Var 'b'
    BinOp: add

    >>> print((c * c).skb.describe_steps())
    Var 'a'
    Var 'b'
    BinOp: add -> _2
    Load _2 (BinOp: add)
    BinOp: mul
    """


#
# traversals of graphs that are not trees
#
# When a node is used several times, the DataOp's graph is a DAG rather than a
# tree. The traversals must not explore such a node (and therefore its whole
# subgraph) once per parent: that used to take a time exponential in the depth
# of the graph, see https://github.com/skrub-data/skrub/issues/2288
#


def _reuse_chain(depth=10):
    """
    A DataOp in which each step uses the previous result 3 times.

    The resulting graph has a few dozen nodes but the number of paths from the
    root to the leaf is 3 ** depth.

    We turn off eager_data_ops so that building the DataOp does not run the
    checks (which are themselves traversals) and does not compute previews:
    no result is cached in the nodes and `evaluate()` has to walk the whole graph.
    """

    @skrub.deferred
    def _halve(x):
        return x / 2

    with skrub.config_context(eager_data_ops=False):
        data_op = skrub.var("v", 8.0) * skrub.choose_float(1.0, 2.0, name="scale")
        for _ in range(depth):
            data_op = (data_op > 1).skb.if_else(_halve(data_op), data_op)
    return data_op


def _count_handle_data_op(monkeypatch, traversal_class, func):
    """Call func() and return the number of handle_data_op calls."""
    original = traversal_class.handle_data_op
    n_calls = 0

    def counting(self, data_op):
        nonlocal n_calls
        n_calls += 1
        return (yield from original(self, data_op))

    monkeypatch.setattr(traversal_class, "handle_data_op", counting)
    func()
    return n_calls


@pytest.mark.parametrize(
    "traversal_class_name, func",
    [
        pytest.param(name, func, id=name)
        for (name, func) in [
            ("_Evaluator", lambda d: _evaluation.evaluate(d, clear=True)),
            ("_Printer", _evaluation.describe_steps),
            ("_Cloner", _evaluation.clone),
            ("_Graph", _evaluation.graph),
            ("_ChoiceGraph", _evaluation.choice_graph),
            ("_ChoiceEvaluator", _evaluation.eval_choices),
            ("_FindNode", lambda d: (_evaluation.find_X(d), _evaluation.find_y(d))),
            ("_FindConflicts", _evaluation.find_conflicts),
            ("_FindArg", lambda d: _evaluation.find_arg(d, lambda arg: False)),
            ("_FindFirstApply", _evaluation.find_first_apply),
        ]
    ],
)
def test_traversals_do_not_re_explore_shared_nodes(
    traversal_class_name, func, monkeypatch
):
    data_op = _reuse_chain()
    n_nodes = len(_evaluation.nodes(data_op))
    n_calls = _count_handle_data_op(
        monkeypatch, getattr(_evaluation, traversal_class_name), lambda: func(data_op)
    )
    # the traversal actually happened, and the number of visits is
    # proportional to the number of nodes rather than to the number of paths.
    assert 0 < n_calls <= 10 * n_nodes


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(func, id=name)
        for (name, func) in [
            ("evaluate", lambda d: _evaluation.evaluate(d, clear=True)),
            ("describe_steps", _evaluation.describe_steps),
            ("clone", _evaluation.clone),
            ("graph", _evaluation.graph),
            ("choice_graph", _evaluation.choice_graph),
            ("eval_choices", _evaluation.eval_choices),
            ("find_X", _evaluation.find_X),
            ("find_conflicts", _evaluation.find_conflicts),
            (
                "find_arg",
                # the default skip_types would skip the Value node that
                # contains the cycle
                lambda d: _evaluation.find_arg(d, lambda arg: False, skip_types=()),
            ),
            ("find_first_apply", _evaluation.find_first_apply),
        ]
    ],
)
def test_circular_references_detected_by_all_traversals(func):
    # Caching the results of the nodes we have already visited must not prevent
    # the detection of cycles (the check happens when a computation is pushed on
    # the stack, i.e. before the cache is looked up).
    value = {}
    value["a"] = [0, {"b": value}]
    with skrub.config_context(eager_data_ops=False):
        data_op = skrub.as_data_op(value)
    with pytest.raises(ValueError, match="DataOps cannot contain circular references"):
        func(data_op)


def test_graph_records_all_the_edges_of_a_shared_node():
    a = skrub.var("a", 1)
    shared = a + 1
    graph = _evaluation.graph(shared * shared)
    assert len(graph["nodes"]) == 3
    # the multiplication has 1 (deduplicated) child and 'shared' is not
    # duplicated even though it is visited twice
    assert graph["children"] == {2: [1], 1: [0]}
    assert graph["parents"] == {1: [2], 0: [1]}


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
