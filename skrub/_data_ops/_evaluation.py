# Utilities to manipulate DataOps: evaluating, cloning, building parameter
# grids etc.
#
# _DataOpTraversal provides the logic for performing a depth-first traversal of
# the computation graph. Subclasses redefine the appropriate methods to provide
# different functionality such as evaluating and cloning.

import copy
import functools
import inspect
import time
import types
import typing
import warnings
from collections import defaultdict
from types import SimpleNamespace

from sklearn.base import BaseEstimator
from sklearn.base import clone as skl_clone

from . import _choosing
from ._data_ops import (
    Apply,
    DataOp,
    Value,
    Var,
)
from ._utils import NULL, X_NAME, Y_NAME, simple_repr

_BUILTIN_SEQ = (list, tuple, set, frozenset)

_BUILTIN_MAP = (dict,)


def _as_gen(f):
    """Turn a regular function into a generator function."""
    if inspect.isgeneratorfunction(f):
        return f

    @functools.wraps(f)
    def g(*args, **kwargs):
        if False:
            yield
        return f(*args, **kwargs)

    return g


class _Computation:
    # Running computations (partially evaluated nodes) on the evaluator's stack
    # are wrapped in those objects that keep track of their target (the object
    # that they are evaluating). This allows inspecting the stack to detect if
    # the same object appears twice, which indicates a circular reference in
    # which case we raise an exception.
    #
    # For example this raises a CircularReferenceError instead of falling in an
    # infinite loop:
    #
    # >>> d = {}
    # >>> d['oops!'] = d
    # >>> skrub.as_data_op(d).skb.eval()

    def __init__(self, target_id, generator):
        self.target_id = target_id
        self.generator = generator


class CircularReferenceError(ValueError):
    """Error raised when the DataOp computation graph contains a cycle."""


class _CurrentNodeDuration:
    """
    How much time has been spent evaluating the current node.

    A `_DataOpTraversal.handle_*()` method can yield an instance of this class
    to obtain the time that has been spent so far on the node that it is
    handling.

    The result counts the time (in seconds) spent on the node itself, excluding
    any time spent on evaluating its children, since the start of the
    `_DataOpTraversal.run()` call.

    The result is the value of the `yield _CurrentNodeDuration()` expression.
    """


class _DataOpTraversal:
    """Base class for objects that manipulate DataOps."""

    # We avoid the use of recursion which could make skrub code harder to debug
    # and more importantly cause very long and confusing traceback for users
    # when something fails.
    #
    # Instead, the nodes that need to be visited are pushed onto a stack. The
    # generator that visits a node yields the children that need to be visited
    # first. The yielded child gets pushed onto the stack. Once its value has
    # been computed, it gets sent back into the generator that yielded it.
    #
    # This is based on the technique described in "The Python Cookbook", D.
    # Beazley, B. Jones, 3rd edition, chapter 8.22 "Implementing the Visitor
    # Pattern Without Recursion" (We do not use the visitor pattern but the
    # way generators are used for control flow in the chapter).
    #
    # A difference is that we do not only need to evaluate skrub node
    # types but also built-in collections, scikit-learn estimators etc. so we
    # use `return` statements in the generators to unambiguously distinguish
    # computed results from child nodes to evaluate. Another is that because we
    # have a large and fast-evolving collection of node types (DataOpImpl
    # subclasses), the logic for computing their result is kept in the DataOpImpl
    # subclass (in the `compute` or `eval`) method rather than in the
    # evaluator. However that logic is very simple because the task of ensuring
    # children are evaluated first is handled by the evaluator (the
    # _DataOpTraversal subclass).

    def run(self, data_op):
        stack = [data_op]
        last_result = None

        # IDs of nodes that are the target of a _Computation currently on the stack.
        # Used to detect circular references.
        running = set()

        # Total time spent evaluating each node (not counting time spent
        # evaluating its children)
        node_durations = defaultdict(float)

        def push_computation(handler):
            "Replace the top of stack (tos) with a _Computation wrapping handler(tos)."
            top = pop()
            top_id = id(top)
            if top_id in running:
                # If 2 computations targeting the same node are on the stack
                # this node is a descendant of itself: we have a cycle in the
                # computation graph. We raise an exception to avoid an infinite
                # loop.
                raise CircularReferenceError(
                    "Skrub DataOps cannot contain circular references. "
                    f"A cycle was found in this object: {top}"
                )
            generator = handler(top)
            stack.append(_Computation(top_id, generator))
            running.add(top_id)

        def pop():
            "Pop an item off the stack."
            top = stack.pop()
            if isinstance(top, _Computation):
                running.remove(top.target_id)
            return top

        def step():
            "Send the last result into the generator at the top of the stack."
            nonlocal last_result
            top = stack[-1]
            try:
                start = time.monotonic()
                try:
                    new_top = top.generator.send(last_result)
                finally:
                    node_durations[top.target_id] += time.monotonic() - start
            except StopIteration as e:
                # The generator returned. The returned value is in the `value`
                # attribute of the `StopIteration`. We store the result and
                # discard the exhausted generator.
                last_result = e.value
                pop()
            else:
                # The generator yielded a new item to evaluate, we push it on
                # the stack.
                stack.append(new_top)
                last_result = None

        while stack:
            top = stack[-1]
            if isinstance(top, _Computation):
                step()
            elif isinstance(top, _CurrentNodeDuration):
                pop()
                last_result = node_durations[stack[-1].target_id]
            elif isinstance(top, DataOp):
                push_computation(self.handle_data_op)

            # We recurse into built-in collections but not their subclasses (we
            # would not know how to reconstruct a collection from the items'
            # values). Thus we compare types directly rather than using isinstance.
            elif type(top) in _BUILTIN_MAP:
                push_computation(self.handle_mapping)
            elif type(top) in _BUILTIN_SEQ:
                push_computation(self.handle_seq)
            elif type(top) is slice:
                push_computation(self.handle_slice)

            elif isinstance(top, _choosing.BaseChoice):
                push_computation(self.handle_choice)
            elif isinstance(top, _choosing.Match):
                push_computation(self.handle_choice_match)
            elif isinstance(top, BaseEstimator):
                push_computation(self.handle_estimator)
            else:
                push_computation(self.handle_value)

        return last_result

    def handle_data_op(self, data_op):
        impl = data_op._skrub_impl
        evaluated_attributes = {}
        for name in impl._fields:
            attr = getattr(impl, name)
            evaluated_attributes[name] = yield attr
        return self.compute_result(data_op, evaluated_attributes)

    def compute_result(self, data_op, evaluated_attributes):
        # Compute the result for a DataOp, once all the children have already
        # been evaluated.
        return data_op

    def handle_estimator(self, estimator):
        params = yield estimator.get_params()
        estimator = skl_clone(estimator)
        estimator.set_params(**params)
        return estimator

    def handle_choice(self, choice):
        if not isinstance(choice, _choosing.Choice):
            # choice is a BaseNumericChoice
            return _choosing._with_fields(choice)
        new_outcomes = yield choice.outcomes
        return _choosing._with_fields(choice, outcomes=new_outcomes)

    def handle_choice_match(self, choice_match):
        choice = yield choice_match.choice
        mapping = yield choice_match.outcome_mapping
        return _choosing._with_fields(
            choice_match, choice=choice, outcome_mapping=mapping
        )

    @_as_gen
    def handle_value(self, value):
        return value

    def handle_seq(self, seq):
        # Note set and frozenset cannot contain directly DataOps and
        # choices which are not hashable but they could in theory contain an
        # estimator that contains a choice (scikit-learn estimators are
        # hashable regardless of their params) so we evaluate items for those
        # collections as well.
        new_seq = []
        for item in seq:
            value = yield item
            new_seq.append(value)
        return type(seq)(new_seq)

    def handle_mapping(self, mapping):
        new_mapping = {}
        for k, v in mapping.items():
            # note evaluating the keys is not needed because DataOps,
            # choices and matches are not hashable so we do not need to
            # (yield k).
            #
            # In theory, because scikit-learn estimators are (unfortunately)
            # hashable an estimator containing a choice could be a key but that
            # wouldn't make sense and evaluating couldn't help in any case
            # because estimators are hashed and compared by id.
            new_mapping[k] = yield v
        return type(mapping)(new_mapping)

    def handle_slice(self, s):
        return slice((yield s.start), (yield s.stop), (yield s.step))


class _Evaluator(_DataOpTraversal):
    # Class used by the evaluate() function defined in this module to evaluate
    # a DataOp.

    def __init__(self, mode="preview", environment=None, callbacks=()):
        self.mode = mode
        self.environment = {} if environment is None else environment
        self.callbacks = callbacks

    def run(self, data_op):
        self._data_op = data_op
        return super().run(data_op)

    def _fetch(self, data_op):
        """Fetch the result from the cache or environment if possible.

        Raises a KeyError otherwise
        """
        impl = data_op._skrub_impl
        if impl.is_X and X_NAME in self.environment:
            return self.environment[X_NAME]
        if impl.is_y and Y_NAME in self.environment:
            return self.environment[Y_NAME]
        if (
            # if Var, let the usual mechanism fetch the value from the
            # environment and store in results dict. Otherwise override with
            # the provided value.
            not isinstance(impl, Var)
            and impl.name is not None
            and impl.name in self.environment
        ):
            return self.environment[impl.name]
        return impl.results[self.mode]

    def _store(self, data_op, result, duration):
        """Store a result in the cache."""
        data_op._skrub_impl.results[self.mode] = result
        metadata = data_op._skrub_impl.metadata.setdefault(self.mode, {})
        metadata["eval_duration"] = duration

    def handle_data_op(self, data_op):
        try:
            return self._fetch(data_op)
        except KeyError:
            pass
        result = yield from self._eval_data_op(data_op)
        duration = yield _CurrentNodeDuration()
        self._store(data_op, result, duration)
        for cb in self.callbacks:
            cb(data_op, result)
        return result

    def _eval_data_op(self, data_op):
        impl = data_op._skrub_impl
        try:
            if hasattr(impl, "eval"):
                return (
                    yield from impl.eval(mode=self.mode, environment=self.environment)
                )
            return (yield from super().handle_data_op(data_op))
        except Exception as e:
            data_op._skrub_impl.errors[self.mode] = e
            if self.mode == "preview":
                raise
            stack = data_op._skrub_impl.creation_stack_last_line()
            msg = (
                f"Evaluation of node {data_op._skrub_impl} failed. See above for full"
                f" traceback. This node was defined here:\n{stack}"
            )
            if hasattr(e, "add_note"):
                e.add_note(msg)
                raise
            # python < 3.11 : we cannot add note to exception so fall back on chaining.
            # Note this changes the type of exception.
            raise RuntimeError(msg) from e

    def handle_choice(self, choice):
        if choice.name is not None and choice.name in self.environment:
            return self.environment[choice.name]
        if self.mode == "preview":
            return (yield _choosing.get_default(choice))
        outcome = choice.chosen_outcome_or_default()
        return (yield outcome)

    def handle_choice_match(self, choice_match):
        outcome = yield choice_match.choice
        return (yield choice_match.outcome_mapping[outcome])

    def compute_result(self, data_op, evaluated_attributes):
        return data_op._skrub_impl.compute(
            SimpleNamespace(**evaluated_attributes),
            mode=self.mode,
            environment=self.environment,
        )


def _check_environment(environment):
    if environment is None:
        return
    if not isinstance(environment, typing.Mapping):
        raise TypeError(
            "`environment` should be a dictionary of input values, for example: "
            "{'X': features_df, 'other_table_name': other_df, ...}. "
            f"Got object of type: '{type(environment)}'"
        )
    env_contains_data_op, found_node = needs_eval(environment, return_node=True)
    if env_contains_data_op:
        if isinstance(found_node, DataOp):
            description = f"a skrub DataOp: {found_node._skrub_impl!r}"
        else:
            description = f"a skrub choice: {found_node}"
        raise TypeError(
            "The `environment` dict "
            f"contains {description}. This argument should only "
            "contain actual values on which to run the computation."
        )
    # Notes about checking the env keys:
    #
    # - env ⊂ variables: in some cases we could check that there are no extra
    #   keys in `environment`, ie all keys in `environment` correspond to a
    #   name in the DataOp. However in other cases we naturally end up
    #   using a bigger environment than what is needed. For example we want tu
    #   evaluate a sub-DataOp (such as the `mark_as_X()` node), and to do
    #   it we use the environment that was passed to evaluate the full
    #   DataOp. So if we want such a verification it should be a separate
    #   check done at a higher level (eg in the estimators' `fit`, `predict`
    #   etc.) where we know we are not working with a sub-DataOp.
    #
    # - variables ⊂ env: we cannot check that all variables in the DataOp
    #   have a matching key in the `environment`, because depending on the
    #   mode, choices, and result of dynamic conditional DataOps such as
    #   `.skb.if_else()` some variables in the DataOp may not be required
    #   for evaluation and those do not need a value and are allowed to be
    #   missing from the environment. For example if we are doing `predict` the
    #   variables used for the computation of `y` do not need to be provided in
    #   the environment because no `y` is computed or used during `predict`.


# TODO switch position of (mode, environment) -> (environment, mode) to be
# consistent with .skb.eval() in evaluate's params


def evaluate(data_op, mode="preview", environment=None, clear=False, callbacks=()):
    """Evaluate a DataOp.

    Parameters
    ----------
    mode : string
        'preview' or the name of a scikit-learn estimator method such as
        'fit_transform' or 'predict'. The evaluation mode.

    environment : dict
        The dict passed by the user, containing binding for all the variables
        contained in the DataOp, e.g., {'users': ..., 'orders': ...}. May
        contain some additional special keys starting with `_skrub` added by
        skrub to control some aspects of the evaluation, eg `_skrub_X` to
        override the value of the node marked with `mark_as_X()`.

    clear : bool
        Clear the result cache of each DataOpImpl node once it is no longer
        needed. Only the cache for the `mode` evaluation mode is used and
        cleared.

    callbacks : list of functions
        Each will be called, in the provided order, after evaluating each node.
        The signature is callback(data_op, result) where data_op is the DataOp
        that was just evaluated and result is the resulting value.
    """
    _check_environment(environment)
    if clear:
        callbacks = (_cache_pruner(data_op, mode),) + tuple(callbacks)
        clear_results(data_op, mode=mode)
    else:
        callbacks = ()
    try:
        return _Evaluator(mode=mode, environment=environment, callbacks=callbacks).run(
            data_op
        )
    finally:
        if clear:
            clear_results(data_op, mode=mode)


class _Reachable(_DataOpTraversal):
    """
    Find all nodes that are reachable from the root node, stopping at nodes
    that have already been computed.

    This is used to find which cached results are no longer needed and can be
    discarded to free the corresponding memory. For example if we have this
    DataOp: `b = a + a; c = b * b` and we want to evaluate `c`, once `b`
    has been computed we no longer need `a` to compute `c` and we can clear its
    cache.
    """

    def __init__(self, mode):
        self.mode = mode

    def run(self, data_op):
        self._reachable = {}
        super().run(data_op)
        return self._reachable

    def handle_data_op(self, data_op):
        self._reachable[id(data_op)] = data_op
        if self.mode in data_op._skrub_impl.results:
            return data_op
        return (yield from super().handle_data_op(data_op))


def _cache_pruner(data_op, mode):
    all_nodes = nodes(data_op)

    def prune(*args, **kwargs):
        reachable_nodes = _Reachable(mode).run(data_op)
        for node in all_nodes:
            if id(node) not in reachable_nodes:
                node._skrub_impl.results.pop(mode, None)

    return prune


class _Printer(_DataOpTraversal):
    """Helper for `describe_steps()`"""

    def run(self, data_op):
        self._seen = set()
        self._lines = []
        self._cache_used = False
        _ = super().run(data_op)
        if self._cache_used:
            self._lines.append("* Cached, not recomputed")
        return "\n".join(self._lines)

    def compute_result(self, data_op, evaluated_attributes):
        is_seen = id(data_op) in self._seen
        line = simple_repr(data_op)
        if is_seen:
            line = f"( {line} )*"
            self._cache_used = True
        self._lines.append(line)
        self._seen.add(id(data_op))


def describe_steps(data_op):
    return _Printer().run(data_op)


class _Cloner(_DataOpTraversal):
    """Helper for `clone()`."""

    # Some objects may appear several times when we traverse the graph (it is a
    # DAG, not always a tree). We keep track of objects we have already cloned
    # in `replace` which maps original object id to the clone, to avoid
    # creating several clones and thus breaking the graph's structure by
    # turning it into a tree.

    def __init__(self, replace=None, drop_preview_data=False):
        self.replace = replace
        self.drop_preview_data = drop_preview_data

    def run(self, data_op):
        self._replace = {} if self.replace is None else dict(self.replace)
        return super().run(data_op)

    def handle_choice(self, choice):
        if id(choice) in self._replace:
            return self._replace[id(choice)]
        new_choice = yield from super().handle_choice(choice)
        self._replace[id(choice)] = new_choice
        return new_choice

    @_as_gen
    def handle_value(self, value):
        if hasattr(value, "__sklearn_clone__") and not isinstance(
            value.__sklearn_clone__, types.MethodType
        ):
            return copy.deepcopy(value)
        return skl_clone(value, safe=False)

    def compute_result(self, data_op, evaluated_attributes):
        if id(data_op) in self._replace:
            return self._replace[id(data_op)]
        impl = data_op._skrub_impl
        new_impl = impl.__replace__(**evaluated_attributes)
        if isinstance(new_impl, Var) and self.drop_preview_data:
            new_impl.value = NULL
        clone = DataOp(new_impl)
        self._replace[id(data_op)] = clone
        return clone


def clone(data_op, replace=None, drop_preview_data=False):
    """Clone a DataOp.

    Parameters
    ----------
    replace : dict or None
        A dict that maps object ids to the object by which they should be
        replaced in the returned clone.

    drop_preview_data : bool
        Whether to drop the `value` attributes of `skrub.var(name, value)`
        nodes.
    """
    return _Cloner(replace=replace, drop_preview_data=drop_preview_data).run(data_op)


def _unique(seq):
    return list(dict.fromkeys(seq))


def _simplify_graph(graph):
    """Replace python object IDs with generated ids starting from 0, 1, ..."""
    short = {v: i for i, v in enumerate(graph["nodes"].keys())}
    new_nodes = {short[k]: v for k, v in graph["nodes"].items()}
    new_children = {
        short[k]: [short[c] for c in _unique(v)] for k, v in graph["children"].items()
    }
    new_parents = {
        short[k]: [short[p] for p in _unique(v)] for k, v in graph["parents"].items()
    }
    return {"nodes": new_nodes, "children": new_children, "parents": new_parents}


class _Graph(_DataOpTraversal):
    """Helper for `graph()`"""

    def run(self, data_op):
        self._nodes = {}
        self._children = defaultdict(list)
        self._parents = defaultdict(list)
        self._current_data_op = []
        _ = super().run(data_op)
        graph = {
            "nodes": self._nodes,
            "children": dict(self._children),
            "parents": dict(self._parents),
        }
        return _simplify_graph(graph)

    def handle_data_op(self, data_op):
        if self._current_data_op:
            child, parent = id(data_op), id(self._current_data_op[-1])
            self._children[parent].append(child)
            self._parents[child].append(parent)
        self._current_data_op.append(data_op)
        result = yield from super().handle_data_op(data_op)
        self._current_data_op.pop()
        self._nodes[id(data_op)] = data_op
        return result


def graph(data_op):
    """Get a simple representation of a DataOp's structure.

    All the nodes (DataOps) contained in the DataOp are numbered
    starting from 0, 1, ...

    This returns a dict with 3 keys:

    - nodes: maps the ID (0, 1, ...) to the corresponding DataOp object
    - children: maps the ID of a node to the list of IDs of its children.
    - parents:maps the ID of a node to the list of IDs of its parents.

    >>> import pprint
    >>> import skrub
    >>> from skrub._data_ops._evaluation import graph
    >>> a = skrub.var('a')
    >>> b = skrub.var('b')
    >>> c = a + b
    >>> d = c * a
    >>> pprint.pprint(graph(d))
    {'children': {2: [0, 1], 3: [2, 0]},
     'nodes': {0: <Var 'a'>, 1: <Var 'b'>, 2: <BinOp: add>, 3: <BinOp: mul>},
     'parents': {0: [2, 3], 1: [2], 2: [3]}}

    the node 3 (the multiplication) has 2 children: 2 (the addition) and 0
    (variable a)
    """
    return _Graph().run(data_op)


def nodes(data_op):
    return list(graph(data_op)["nodes"].values())


def clear_results(data_op, mode=None):
    for n in nodes(data_op):
        if mode is None:
            n._skrub_impl.results = {}
            n._skrub_impl.errors = {}
            n._skrub_impl.metadata = {}
        else:
            n._skrub_impl.results.pop(mode, None)
            n._skrub_impl.errors.pop(mode, None)
            n._skrub_impl.metadata.pop(mode, None)


def _choice_display_names(choices):
    """
    Get display names (eg for parallel coord plots) for all choices in a
    DataOp.

    When the choice is given an explicit `name` by the user that is used,
    otherwise a shorted repr + number suffix to make them unique.
    """
    used = set()
    names = {}

    def add(choice_ids):
        for c_id in choice_ids:
            stem = _choosing.get_display_name(choices[c_id])
            if stem not in used:
                used.add(stem)
                names[c_id] = stem
                continue
            i = 1
            while (numbered := f"{stem}_{i}") in used:
                i += 1
            used.add(numbered)
            names[c_id] = numbered
        return names

    add(c_id for (c_id, c) in choices.items() if c.name is not None)
    add(c_id for (c_id, c) in choices.items() if c.name is None)
    # keep the same order as in choices
    return {c_id: names[c_id] for c_id in choices.keys()}


class _ChoiceGraph(_DataOpTraversal):
    """Helper for `choice_graph()`."""

    def run(self, data_op):
        self._choices = {}
        self._children = defaultdict(list)
        self._current_outcome = [None]

        _ = super().run(data_op)

        short = {choice_id: i for i, choice_id in enumerate(self._choices.keys())}
        self._short_ids = short
        choices = {
            short[choice_id]: choice for choice_id, choice in self._choices.items()
        }
        children = {}
        for outcome_key, outcome_children in self._children.items():
            if outcome_key is None:
                new_key = None
            else:
                choice_id, outcome_idx = outcome_key
                new_key = short[choice_id], outcome_idx
            new_outcome_children = [
                short[child_id] for child_id in _unique(outcome_children)
            ]
            children[new_key] = new_outcome_children
        # - choices:
        #     choice's short id (1, 2, ...) to BaseChoice instance
        # - children:
        #     (choice short id, outcome index) to list of child choices' short ids
        return {
            "choices": choices,
            "children": children,
            "choice_display_names": _choice_display_names(choices),
        }

    def handle_choice(self, choice):
        self._children[self._current_outcome[-1]].append(id(choice))
        if not isinstance(choice, _choosing.Choice):
            self._choices[id(choice)] = choice
            return choice
        for outcome_idx, outcome in enumerate(choice.outcomes):
            self._current_outcome.append((id(choice), outcome_idx))
            yield outcome
            self._current_outcome.pop()
        self._choices[id(choice)] = choice
        return choice

    def handle_choice_match(self, choice_match):
        yield choice_match.choice
        for outcome_idx, outcome in enumerate(choice_match.choice.outcomes):
            self._current_outcome.append((id(choice_match.choice), outcome_idx))
            yield choice_match.outcome_mapping[outcome]
            self._current_outcome.pop()
        return choice_match


def choice_graph(data_op, check_Xy=True):
    """The graph of all the choices in a DataOp.

    All BaseChoice objects found are numbered from 0, 1, ...
    Those IDs are used to describe the nested choices structure and to define
    the parameter names for parameter grid, get_params, set_params.

    This is different from `graph()` which collects nodes (DataOpImpl objects);
    `choice_graph` is for inspecting `choose_from` objects and build parameter
    grids.

    Parameters
    ----------
    data_op : the DataOp to inspect

    check_Xy : bool
        Choices upstream of X and y, ie upstream of the train/test split cannot
        be tuned (we would be selecting the easiest problem rather than the
        best learner). If `check_Xy` is True, when such choices exist we emit a
        warning and list them in the result under the `"Xy_choices"` key, which
        is then used to clamp those choices to their default value when
        building the parameter grid.

    Returns
    -------

    A dict with several keys:

     - choices : maps choice ID (0, 1, ...) to the choice object
     - Xy_choices : set of IDs of choices upstream of X and y
     - choice_display_names: maps choice IDs to the name to display for that
       choice in parallel coord plots, human-readable representations of the
       param grid etc.
     - children :
       Helps identify nested choices. Choice instances (created by
       `choose_from`) can have arbitrary objects (such as DataOps,
       scikit-learn estimators, choices) as their outcomes. Some of those
       outcomes may themselves contain choices. In this case the pair (choice
       ID, outcome index) is added as a key in the `children` mapping; the
       value is the list of IDs of the choices contained in the outcome.
       Choices at the top level (that don't have a parent) are listed in
       children under the key `None`

    For example :

    >>> from pprint import pprint
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.dummy import DummyRegressor
    >>> import skrub
    >>> from skrub import choose_from, choose_float
    >>> from skrub._data_ops._evaluation import choice_graph

    >>> e = choose_from(
    ...     [Ridge(alpha=choose_float(0.1, 1.0, name="alpha")), DummyRegressor()],
    ...     name="regressor",
    ... ).as_data_op()

    >>> pprint(choice_graph(e))
    {'Xy_choices': set(),
     'children': {None: [1], (1, 0): [0]},
     'choice_display_names': {0: 'alpha', 1: 'regressor'},
     'choices': {0: choose_float(0.1, 1.0, name='alpha'),
                 1: choose_from([Ridge(alpha=choose_float(0.1, 1.0, name='alpha')), DummyRegressor()], name='regressor')}}

    The choice 'alpha' (0) is contained in the first outcome of the choice
    'regressor' (1). Therefore 'children' contains
    (1,                 0):                            [0]
     ^                  ^                               ^
     ID of 'regressor'  index of first outcome (Ridge)  ID of 'alpha'

    The choice 'regressor' has no parents so it is listed as a child of `None`
    """  # noqa: E501
    full_builder = _ChoiceGraph()
    full_graph = full_builder.run(data_op)
    if not check_Xy:
        return full_graph
    # identify which choices are used before the nodes marked as X or y. Those
    # choices cannot be tuned (they are needed before the cv loop starts) so
    # they will be clamped to a single value (the chosen outcome if that has been
    # set otherwise the default)
    Xy_choices = set()
    for node in [find_X(data_op), find_y(data_op)]:
        if node is not None:
            node_graph = _ChoiceGraph().run(node)
            Xy_choices.update(
                {
                    # convert from python id to short id
                    full_builder._short_ids[id(c)]
                    for c in node_graph["choices"].values()
                }
            )
    if Xy_choices:
        warnings.warn(
            "The following choices are used in the construction of X or y, "
            "so their value cannot be tuned because they are needed outside "
            "of the cross-validation loop. They will be clamped to their "
            f"default value: {[full_graph['choices'][k] for k in Xy_choices]}"
        )
    full_graph["Xy_choices"] = Xy_choices
    return full_graph


def choices(data_op):
    return choice_graph(data_op, check_Xy=False)["choices"]


def check_choices_before_Xy(data_op):
    """Emit a warning if there are hyperparameters upstream of X or y."""
    choice_graph(data_op)


def _expand_grid(graph, grid):
    def choice_range(choice_id):
        # The range of possible values for a choice.
        # for numeric choices it is the object itself and for Choices the range
        # of possible outcome indices.
        # if the choice is used in X or y, it is clamped to a single value
        choice = graph["choices"][choice_id]
        if choice_id in graph["Xy_choices"]:
            if isinstance(choice, _choosing.Choice):
                return [choice.chosen_outcome_idx or 0]
            else:
                return [choice.default() if (o := choice.chosen_outcome) is None else o]
        else:
            if isinstance(choice, _choosing.Choice):
                return list(range(len(choice.outcomes)))
            else:
                return choice

    def has_children(choice_id):
        # if any of the outcomes in a choice contains another choice. in this
        # case it needs to be on a separate subgrid.
        choice = graph["choices"][choice_id]
        if not isinstance(choice, _choosing.Choice):
            return False
        for outcome_idx in choice_range(choice_id):
            if graph["children"].get((choice_id, outcome_idx), None):
                return True
        return False

    # extract
    if None not in graph["children"]:
        return [grid]
    for choice_id in graph["children"][None]:
        if not has_children(choice_id):
            grid[choice_id] = choice_range(choice_id)
    # split
    remaining = [c_id for c_id in graph["children"][None] if c_id not in grid]
    if not remaining:
        return [grid]
    choice_id = remaining[0]
    subgrids = []
    for outcome_idx in choice_range(choice_id):
        new_subgrid = grid.copy()
        graph = graph.copy()
        graph["children"][None] = (
            graph["children"].get((choice_id, outcome_idx), []) + remaining[1:]
        )
        new_subgrid[choice_id] = [outcome_idx]
        new_subgrid = _expand_grid(graph, new_subgrid)
        subgrids.extend(new_subgrid)
    return subgrids


def param_grid(data_op):
    """
    Build the parameter grid (for GridSearchCV and RandomizedSearchCV) for a
    DataOp.
    """
    graph = choice_graph(data_op)
    return _expand_grid(graph, {})


def get_params(data_op):
    data_op_choices = choices(data_op)
    params = {}
    for k, v in data_op_choices.items():
        if isinstance(v, _choosing.Choice):
            params[k] = getattr(v, "chosen_outcome_idx", None)
        else:
            params[k] = getattr(v, "chosen_outcome", None)
    return params


def set_params(data_op, params):
    data_op_choices = choices(data_op)
    for k, v in params.items():
        target = data_op_choices[k]
        if isinstance(target, _choosing.Choice):
            target.chosen_outcome_idx = v
        else:
            target.chosen_outcome = v


class _ChosenOrDefaultOutcomes(_DataOpTraversal):
    """Helper for `chosen_or_default_outcomes`."""

    def run(self, data_op):
        self.chosen = {}
        self.results = {}
        _ = super().run(data_op)
        return self.chosen

    def handle_choice(self, choice):
        if id(choice) in self.results:
            return self.results[id(choice)]
        if not isinstance(choice, _choosing.Choice):
            # We have a NumericChoice, the outcome is simply a number
            outcome = choice.chosen_outcome_or_default()
            self.chosen[id(choice)] = outcome
            self.results[id(choice)] = outcome
            return outcome
        # We have a Choice and need to visit the chosen outcome (it may contain
        # further choices).
        idx = choice.chosen_outcome_idx or 0
        self.chosen[id(choice)] = idx
        outcome = choice.outcomes[idx]
        result = yield outcome
        self.results[id(choice)] = result
        return result

    def handle_choice_match(self, choice_match):
        outcome = yield choice_match.choice
        return (yield choice_match.outcome_mapping[outcome])


def chosen_or_default_outcomes(data_op):
    """Get the selected or default outcomes for choices in the DataOp.

    Return a mapping from the choice's ID (0, 1, ... -- see `choice_graph`) to
    the corresponding outcome.

    When the choice outcome has been set with set_params, that is used,
    otherwise the choice's default.

    Importantly, choices that are not used in the set (or default)
    configuration do not appear in the result.
    This is why a different traversal scheme is needed than in `choice_graph`,
    because we only explore subgraphs corresponding to the selected outcomes.

    >>> from pprint import pprint
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.dummy import DummyRegressor
    >>> from skrub import choose_from, choose_float
    >>> from skrub._data_ops._evaluation import chosen_or_default_outcomes, choices

    >>> e = choose_from(
    ...     [DummyRegressor(), Ridge(alpha=choose_float(0.1, 1.0, name="alpha"))],
    ...     name="regressor",
    ... ).as_data_op()

    All the choices found in the DataOp: mapping from choice ID to the
    corresponding choice object:

    >>> pprint(choices(e))
    {0: choose_float(0.1, 1.0, name='alpha'),
     1: choose_from([DummyRegressor(), Ridge(alpha=choose_float(0.1, 1.0, name='alpha'))], name='regressor')}

    In the default configuration, the 'regressor' chooses the DummyRegressor. So the
    'alpha' choice is not used. It does not appear in the
    `chosen_or_default_outcomes` result:

    >>> pprint(chosen_or_default_outcomes(e))
    {1: 0}

    Here we only see that choice 'regressor' (ID 1) in chooses its first outcome
    (index 0), and the choice 'alpha' (ID 0) does not appear.
    """  # noqa: E501
    data_op_choices = choices(data_op)
    short_ids = {id(c): k for k, c in data_op_choices.items()}
    outcomes = _ChosenOrDefaultOutcomes().run(data_op)
    return {short_ids[k]: v for k, v in outcomes.items()}


class _Found(Exception):
    def __init__(self, value):
        self.value = value


class _FindNode(_DataOpTraversal):
    def __init__(self, predicate=None):
        self.predicate = predicate

    def handle_data_op(self, e):
        if self.predicate is None or self.predicate(e):
            raise _Found(e)
        yield from super().handle_data_op(e)

    def handle_choice(self, choice):
        if self.predicate is None or self.predicate(choice):
            raise _Found(choice)
        yield from super().handle_choice(choice)


def find_node(obj, predicate=None):
    """Find a DataOp or choice in the graph according to `predicate`.

    The first one that is found according to a deterministic traversal order is
    returned, None is returned if no such node is found.
    """
    try:
        _FindNode(predicate).run(obj)
    except _Found as e:
        return e.value
    return None


def find_X(data_op):
    return find_node(data_op, lambda e: isinstance(e, DataOp) and e._skrub_impl.is_X)


def find_y(data_op):
    return find_node(data_op, lambda e: isinstance(e, DataOp) and e._skrub_impl.is_y)


def find_node_by_name(data_op, name):
    def pred(obj):
        if isinstance(obj, DataOp):
            return obj._skrub_impl.name == name
        return getattr(obj, "name", None) == name

    return find_node(data_op, pred)


def needs_eval(obj, return_node=False):
    """
    Whether a python object contains any object that requires evaluation such
    as a DataOp or a choice.
    """
    try:
        node = find_node(obj)
    except CircularReferenceError:
        node = None
    needs = node is not None
    if return_node:
        return needs, node
    return needs


class _FindConflicts(_DataOpTraversal):
    """Find duplicate names or if 2 nodes are marked as X or y."""

    def __init__(self):
        self._names = {}
        self._x = {}
        self._y = {}

    def handle_data_op(self, e):
        self._add(
            e,
            getattr(e._skrub_impl, "name", None),
            e._skrub_impl.is_X,
            e._skrub_impl.is_y,
        )
        yield from super().handle_data_op(e)

    def handle_choice(self, choice):
        self._add(choice, getattr(choice, "name", None), False, False)
        yield from super().handle_choice(choice)

    def _conflict_error_message(self, conflict):
        first, second = conflict["nodes"]
        if conflict["reason"] == "is_X":
            return (
                "Only one node can be marked with `mark_as_X()`. "
                "2 different objects were marked as X:\n"
                f"first object that used `.mark_as_X()`:\n{first}\n"
                f"second object that used `.mark_as_X()`:\n{second}"
            )
        if conflict["reason"] == "is_y":
            return (
                "Only one node can be marked with `mark_as_y()`. "
                "2 different objects were marked as y:\n"
                f"first object that used `.mark_as_y()`:\n{first}\n"
                f"second object that used `.mark_as_y()`:\n{second}"
            )
        assert conflict["reason"] == "name", conflict["reason"]
        name = conflict["name"]
        msg = (
            f"Choice and node names must be unique. The name {name!r} was used "
            "for 2 different objects:\n"
            f"first object using the name {name!r}:\n{first}\n"
            f"second object using the name {name!r}:\n{second}"
        )
        if repr(first) == repr(second):
            msg += (
                "\nIs it possible that you accidentally added a transformation twice, "
                "for example by re-running a Jupyter notebook cell that rebinds a "
                "Python variable to a new skrub data_op "
                "(eg `data_op = data_op.skb.apply(...)`)?"
            )
        return msg

    def _add_to_dict(self, d, key, val, reason):
        if key is None:
            return
        other = d.get(key, None)
        if other is None:
            d[key] = val
            return
        if other is val:
            return
        conflict = {"name": key, "nodes": (other, val), "reason": reason}
        conflict["message"] = self._conflict_error_message(conflict)
        raise _Found(conflict)

    def _add(self, obj, name, is_X, is_y):
        if is_X:
            self._add_to_dict(self._x, "X", obj, "is_X")
        if is_y:
            self._add_to_dict(self._y, "y", obj, "is_y")
        self._add_to_dict(self._names, name, obj, "name")


def find_conflicts(data_op):
    """
    We use a function that returns the conflicts, rather than raises an
    exception, because we want the exception to be raised higher in the call
    stack (in ``_data_ops._check_data_op``) so that the user sees the line in
    their code that created a problematic DataOp easily in the traceback.
    """
    try:
        _FindConflicts().run(data_op)
    except _Found as e:
        return e.value
    return None


class _FindArg(_DataOpTraversal):
    def __init__(self, predicate, skip_types=(Var, Value)):
        self.predicate = predicate
        self.skip_types = skip_types

    def handle_data_op(self, data_op):
        if isinstance(data_op._skrub_impl, self.skip_types):
            return data_op
        return (yield from super().handle_data_op(data_op))

    def handle_value(self, value):
        if self.predicate(value):
            raise _Found(value)
        return (yield from super().handle_value(value))


def find_arg(data_op, predicate, skip_types=(Var, Value)):
    # Find a node while ignoring certain DataOp types, used by
    # _data_ops._find_dataframe to detect when someone passed an actual
    # DataFrame instead of a DataOp to a DataOp's method or to a
    # deferred function.
    try:
        _FindArg(predicate, skip_types=skip_types).run(data_op)
    except _Found as e:
        return e.value
    return None


class _FindFirstApply(_DataOpTraversal):
    def handle_choice(self, choice):
        return (yield choice.chosen_outcome_or_default())

    def handle_data_op(self, data_op):
        if isinstance(data_op._skrub_impl, Apply):
            raise _Found(data_op)
        return (yield from super().handle_data_op(data_op))


def find_first_apply(data_op):
    """Find the Apply() node closest to the DataOp's root.

    This is assumed to be the final/supervised learner and inspected in
    _estimator to determine its nature (regressor, classifier, transformer) and
    attributes (eg whether it has predict_proba())
    """
    try:
        _FindFirstApply().run(data_op)
    except _Found as first:
        return first.value
    return None


def supported_modes(data_op):
    """The evaluation modes that the final estimator supports."""
    first = find_first_apply(data_op)
    if first is None:
        return ["preview", "fit", "fit_transform", "transform"]
    return first._skrub_impl.supported_modes()
