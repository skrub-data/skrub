import copy
import functools
import inspect
import types
import typing
import warnings
from collections import defaultdict
from types import SimpleNamespace

from sklearn.base import BaseEstimator
from sklearn.base import clone as skl_clone

from . import _choosing
from ._expressions import (
    Apply,
    Expr,
    IfElse,
    Match,
    Value,
    Var,
)
from ._utils import NULL, X_NAME, Y_NAME, simple_repr

__all__ = [
    "evaluate",
    "clone",
    "find_X",
    "find_y",
    "find_node_by_name",
    "graph",
    "nodes",
    "clear_results",
    "describe_steps",
    "get_params",
    "set_params",
    "check_choices_before_Xy",
]

_BUILTIN_SEQ = (list, tuple, set, frozenset)

_BUILTIN_MAP = (dict,)


def _as_gen(f):
    if inspect.isgeneratorfunction(f):
        return f

    @functools.wraps(f)
    def g(*args, **kwargs):
        if False:
            yield
        return f(*args, **kwargs)

    return g


class _Computation:
    def __init__(self, target, generator):
        self.target_id = id(target)
        self.generator = generator


class CircularReferenceError(ValueError):
    pass


class _ExprTraversal:
    def run(self, expr):
        stack = [expr]
        last_result = None

        def push(handler):
            top = stack.pop()
            generator = handler(top)
            stack.append(_Computation(top, generator))

        while stack:
            top = stack[-1]
            try:
                if isinstance(top, _Computation):
                    new_top = top.generator.send(last_result)
                    if id(new_top) in {
                        c.target_id for c in stack if isinstance(c, _Computation)
                    }:
                        raise CircularReferenceError(
                            "Skrub expressions cannot contain circular references. "
                            f"A circular reference was found in this object: {new_top}"
                        )
                    stack.append(new_top)
                    last_result = None
                elif isinstance(top, Expr):
                    push(self.handle_expr)
                elif isinstance(top, _BUILTIN_MAP):
                    push(self.handle_mapping)
                elif isinstance(top, _BUILTIN_SEQ):
                    push(self.handle_seq)
                elif isinstance(top, slice):
                    push(self.handle_slice)
                elif isinstance(top, _choosing.BaseChoice):
                    push(self.handle_choice)
                elif isinstance(top, _choosing.Match):
                    push(self.handle_choice_match)
                elif isinstance(top, BaseEstimator):
                    push(self.handle_estimator)
                else:
                    push(self.handle_value)
            except StopIteration as e:
                last_result = e.value
                stack.pop()
        return last_result

    def handle_expr(self, expr, attributes_to_evaluate=None):
        impl = expr._skrub_impl
        evaluated_attributes = {}
        if attributes_to_evaluate is None:
            attributes_to_evaluate = impl._fields
        for name in attributes_to_evaluate:
            attr = getattr(impl, name)
            evaluated_attributes[name] = yield attr
        return self.compute_result(expr, evaluated_attributes)

    def compute_result(self, expr, evaluated_attributes):
        return expr

    def handle_estimator(self, estimator):
        params = yield estimator.get_params()
        estimator = skl_clone(estimator)
        estimator.set_params(**params)
        return estimator

    def handle_choice(self, choice):
        if not isinstance(choice, _choosing.Choice):
            # choice is a BaseNumericChoice
            return choice
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
        # Note set and frozenset cannot contain directly expressions and
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
            # note evaluating the keys is not needed because expressions,
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


class _Evaluator(_ExprTraversal):
    def __init__(self, mode="preview", environment=None, callbacks=()):
        self.mode = mode
        self.environment = {} if environment is None else environment
        self.callbacks = callbacks

    def run(self, expr):
        self._expr = expr
        return super().run(expr)

    def _fetch(self, expr):
        """Fetch the result from the cache or environment if possible.

        Raises a KeyError otherwise
        """
        impl = expr._skrub_impl
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

    def _store(self, expr, result):
        """Store a result in the cache."""
        expr._skrub_impl.results[self.mode] = result

    def handle_expr(self, expr):
        try:
            return self._fetch(expr)
        except KeyError:
            pass
        impl = expr._skrub_impl
        if isinstance(impl, IfElse):
            result = yield from self._handle_if_else(expr)
        elif isinstance(impl, Match):
            result = yield from self._handle_match(expr)
        else:
            result = yield from self._handle_expr_default(expr)
        self._store(expr, result)
        for cb in self.callbacks:
            cb(expr, result)
        return result

    def _handle_if_else(self, expr):
        impl = expr._skrub_impl
        cond = yield impl.condition
        if cond:
            return (yield impl.value_if_true)
        else:
            return (yield impl.value_if_false)

    def _handle_match(self, expr):
        impl = expr._skrub_impl
        query = yield impl.query
        if impl.has_default():
            target = impl.targets.get(query, impl.default)
        else:
            target = impl.targets[query]
        return (yield target)

    def _handle_expr_default(self, expr):
        return (
            yield from super().handle_expr(
                expr, expr._skrub_impl.fields_required_for_eval(self.mode)
            )
        )

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

    def compute_result(self, expr, evaluated_attributes):
        try:
            return expr._skrub_impl.compute(
                SimpleNamespace(**evaluated_attributes),
                mode=self.mode,
                environment=self.environment,
            )
        except Exception as e:
            expr._skrub_impl.errors[self.mode] = e
            if self.mode == "preview":
                raise
            stack = expr._skrub_impl.creation_stack_last_line()
            msg = (
                f"Evaluation of node {expr._skrub_impl} failed. See above for full"
                f" traceback. This node was defined here:\n{stack}"
            )
            if hasattr(e, "add_note"):
                e.add_note(msg)
                raise
            # python < 3.11 : we cannot add note to exception so fall back on chaining
            # note this changes the type of exception
            raise RuntimeError(msg) from e


def _check_environment(environment):
    if environment is None:
        return
    if not isinstance(environment, typing.Mapping):
        raise TypeError(
            "`environment` should be a dictionary of input values, for example: "
            "{'X': features_df, 'other_table_name': other_df, ...}. "
            f"Got object of type: '{type(environment)}'"
        )
    env_contains_expr, found_node = needs_eval(environment, return_node=True)
    if env_contains_expr:
        if isinstance(found_node, Expr):
            description = f"a skrub expression: {found_node._skrub_impl!r}"
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
    #   name in the expression. However in other cases we naturally end up
    #   using a bigger environment than what is needed. For example we want tu
    #   evaluate a sub-expression (such as the `mark_as_X()` node), and to do
    #   it we use the environment that was passed to evaluate the full
    #   expression. So if we want such a verification it should be a separate
    #   check done at a higher level (eg in the estimators' `fit`, `predict`
    #   etc.) where we know we are not working with a sub-expression.
    #
    # - variables ⊂ env: we cannot check that all variables in the expression
    #   have a matching key in the `environment`, because depending on the
    #   mode, choices, and result of dynamic conditional expressions such as
    #   `.skb.if_else()` some variables in the expression may not be required
    #   for evaluation and those do not need a value and are allowed to be
    #   missing from the environment. For example if we are doing `predict` the
    #   variables used for the computation of `y` do not need to be provided in
    #   the environment because no `y` is computed or used during `predict`.


# TODO switch position of (mode, environment) -> (environment, mode) to be
# consistent with .skb.eval() in evaluate's params


def evaluate(expr, mode="preview", environment=None, clear=False, callbacks=()):
    requested_mode = mode
    mode = "fit_transform" if requested_mode == "fit" else requested_mode
    _check_environment(environment)
    if clear:
        callbacks = (_cache_pruner(expr, mode),) + tuple(callbacks)
        clear_results(expr, mode=mode)
    else:
        callbacks = ()
    try:
        result = _Evaluator(
            mode=mode, environment=environment, callbacks=callbacks
        ).run(expr)
        return expr if requested_mode == "fit" else result
    finally:
        if clear:
            clear_results(expr, mode=mode)


class _Reachable(_ExprTraversal):
    def __init__(self, mode):
        self.mode = mode

    def run(self, expr):
        self._reachable = {}
        super().run(expr)
        return self._reachable

    def handle_expr(self, expr):
        self._reachable[id(expr)] = expr
        if self.mode in expr._skrub_impl.results:
            return expr
        return (yield from super().handle_expr(expr))


def _cache_pruner(expr, mode):
    all_nodes = nodes(expr)

    def prune(*args, **kwargs):
        reachable_nodes = _Reachable(mode).run(expr)
        for node in all_nodes:
            if id(node) not in reachable_nodes:
                node._skrub_impl.results.pop(mode, None)

    return prune


class _Printer(_ExprTraversal):
    def run(self, expr):
        self._seen = set()
        self._lines = []
        self._cache_used = False
        _ = super().run(expr)
        if self._cache_used:
            self._lines.append("* Cached, not recomputed")
        return "\n".join(self._lines)

    def compute_result(self, expr, evaluated_attributes):
        is_seen = id(expr) in self._seen
        line = simple_repr(expr)
        if is_seen:
            line = f"( {line} )*"
            self._cache_used = True
        self._lines.append(line)
        self._seen.add(id(expr))


def describe_steps(expr):
    return _Printer().run(expr)


class _Cloner(_ExprTraversal):
    def __init__(self, replace=None, drop_preview_data=False):
        self.replace = replace
        self.drop_preview_data = drop_preview_data

    def run(self, expr):
        self._replace = {} if self.replace is None else dict(self.replace)
        return super().run(expr)

    def handle_choice(self, choice):
        if id(choice) in self._replace:
            return self._replace[id(choice)]
        new_choice = yield from super().handle_choice(choice)
        if not isinstance(new_choice, _choosing.Choice):
            new_choice = _choosing._with_fields(choice)
        self._replace[id(choice)] = new_choice
        return new_choice

    @_as_gen
    def handle_value(self, value):
        if hasattr(value, "__sklearn_clone__") and not isinstance(
            value.__sklearn_clone__, types.MethodType
        ):
            return copy.deepcopy(value)
        return skl_clone(value, safe=False)

    def compute_result(self, expr, evaluated_attributes):
        if id(expr) in self._replace:
            return self._replace[id(expr)]
        impl = expr._skrub_impl
        new_impl = impl.__replace__(**evaluated_attributes)
        if isinstance(new_impl, Var) and self.drop_preview_data:
            new_impl.value = NULL
        clone = Expr(new_impl)
        self._replace[id(expr)] = clone
        return clone


def clone(expr, replace=None, drop_preview_data=False):
    return _Cloner(replace=replace, drop_preview_data=drop_preview_data).run(expr)


def _unique(seq):
    return list(dict.fromkeys(seq))


def _simplify_graph(graph):
    short = {v: i for i, v in enumerate(graph["nodes"].keys())}
    new_nodes = {short[k]: v for k, v in graph["nodes"].items()}
    new_children = {
        short[k]: [short[c] for c in _unique(v)] for k, v in graph["children"].items()
    }
    new_parents = {
        short[k]: [short[p] for p in _unique(v)] for k, v in graph["parents"].items()
    }
    return {"nodes": new_nodes, "children": new_children, "parents": new_parents}


class _Graph(_ExprTraversal):
    def run(self, expr):
        self._nodes = {}
        self._children = defaultdict(list)
        self._parents = defaultdict(list)
        self._current_expr = []
        _ = super().run(expr)
        graph = {
            "nodes": self._nodes,
            "children": dict(self._children),
            "parents": dict(self._parents),
        }
        return _simplify_graph(graph)

    def handle_expr(self, expr):
        if self._current_expr:
            child, parent = id(expr), id(self._current_expr[-1])
            self._children[parent].append(child)
            self._parents[child].append(parent)
        self._current_expr.append(expr)
        result = yield from super().handle_expr(expr)
        self._current_expr.pop()
        self._nodes[id(expr)] = expr
        return result


def graph(expr):
    return _Graph().run(expr)


def nodes(expr):
    return list(graph(expr)["nodes"].values())


def clear_results(expr, mode=None):
    # TODO: create a context manager for clearing results
    for n in nodes(expr):
        if mode is None:
            n._skrub_impl.results = {}
            n._skrub_impl.errors = {}
        else:
            n._skrub_impl.results.pop(mode, None)
            n._skrub_impl.errors.pop(mode, None)


class _ChoiceGraph(_ExprTraversal):
    def run(self, expr):
        self._choices = {}
        self._children = defaultdict(list)
        self._current_outcome = [None]

        _ = super().run(expr)

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
        return {"choices": choices, "children": children}

    def handle_choice(self, choice):
        # unlike during evaluation here we need pre-ordering
        self._children[self._current_outcome[-1]].append(id(choice))
        self._choices[id(choice)] = choice
        if not isinstance(choice, _choosing.Choice):
            return choice
        for outcome_idx, outcome in enumerate(choice.outcomes):
            self._current_outcome.append((id(choice), outcome_idx))
            yield outcome
            self._current_outcome.pop()
        return choice

    def handle_choice_match(self, choice_match):
        yield choice_match.choice
        for outcome_idx, outcome in enumerate(choice_match.choice.outcomes):
            self._current_outcome.append((id(choice_match.choice), outcome_idx))
            yield choice_match.outcome_mapping[outcome]
            self._current_outcome.pop()
        return choice_match


def choices(expr):
    return _ChoiceGraph().run(expr)["choices"]


def choice_graph(expr):
    full_builder = _ChoiceGraph()
    full_graph = full_builder.run(expr)
    # identify which choices are used before the nodes marked as X or y. Those
    # choices cannot be tuned (they are needed before the cv loop starts) so
    # they will be clamped to a single value (the chosen outcome if that has been
    # set otherwise the default)
    Xy_choices = set()
    for node in [find_X(expr), find_y(expr)]:
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


def check_choices_before_Xy(expr):
    """Emit a warning if there are hyperparameters upstream of X or y."""
    choice_graph(expr)


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


def param_grid(expr):
    graph = choice_graph(expr)
    return _expand_grid(graph, {})


def get_params(expr):
    expr_choices = choices(expr)
    params = {}
    for k, v in expr_choices.items():
        if isinstance(v, _choosing.Choice):
            params[k] = getattr(v, "chosen_outcome_idx", None)
        else:
            params[k] = getattr(v, "chosen_outcome", None)
    return params


def set_params(expr, params):
    expr_choices = choices(expr)
    for k, v in params.items():
        target = expr_choices[k]
        if isinstance(target, _choosing.Choice):
            target.chosen_outcome_idx = v
        else:
            target.chosen_outcome = v


class _Found(Exception):
    def __init__(self, value):
        self.value = value


class _FindNode(_ExprTraversal):
    def __init__(self, predicate=None):
        self.predicate = predicate

    def handle_expr(self, e, *args, **kwargs):
        if self.predicate is None or self.predicate(e):
            raise _Found(e)
        yield from super().handle_expr(e, *args, **kwargs)

    def handle_choice(self, choice):
        if self.predicate is None or self.predicate(choice):
            raise _Found(choice)
        yield from super().handle_choice(choice)


def find_node(obj, predicate=None):
    try:
        _FindNode(predicate).run(obj)
    except _Found as e:
        return e.value
    return None


def find_X(expr):
    return find_node(expr, lambda e: isinstance(e, Expr) and e._skrub_impl.is_X)


def find_y(expr):
    return find_node(expr, lambda e: isinstance(e, Expr) and e._skrub_impl.is_y)


def find_node_by_name(expr, name):
    def pred(obj):
        if isinstance(obj, Expr):
            return obj._skrub_impl.name == name
        return getattr(obj, "name", None) == name

    return find_node(expr, pred)


def needs_eval(obj, return_node=False):
    try:
        node = find_node(obj)
    except CircularReferenceError:
        node = None
    needs = node is not None
    if return_node:
        return needs, node
    return needs


class _FindConflicts(_ExprTraversal):
    """Find duplicate names or if 2 nodes are marked as X or y."""

    def __init__(self):
        self._names = {}
        self._x = {}
        self._y = {}

    def handle_expr(self, e, *args, **kwargs):
        self._add(
            e,
            getattr(e._skrub_impl, "name", None),
            e._skrub_impl.is_X,
            e._skrub_impl.is_y,
        )
        yield from super().handle_expr(e, *args, **kwargs)

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
        return (
            f"Choice and node names must be unique. The name {name!r} was used "
            "for 2 different objects:\n"
            f"first object using the name {name!r}:\n{first}\n"
            f"second object using the name {name!r}:\n{second}"
        )

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


def find_conflicts(expr):
    """
    We use a function that returns the conflicts, rather than raises an
    exception, because we want the exception to be raised higher in the call
    stack (in ``_expressions._check_expr``) so that the user sees the line in
    their code that created a problematic expression easily in the traceback.
    """
    try:
        _FindConflicts().run(expr)
    except _Found as e:
        return e.value
    return None


class _FindArg(_ExprTraversal):
    def __init__(self, predicate, skip_types=(Var, Value)):
        self.predicate = predicate
        self.skip_types = skip_types

    def handle_expr(self, expr, **kwargs):
        if isinstance(expr._skrub_impl, self.skip_types):
            return expr
        return (yield from super().handle_expr(expr, **kwargs))

    def handle_value(self, value):
        if self.predicate(value):
            raise _Found(value)
        return (yield from super().handle_value(value))


def find_arg(expr, predicate, skip_types=(Var, Value)):
    try:
        _FindArg(predicate, skip_types=skip_types).run(expr)
    except _Found as e:
        return e.value
    return None


class _FindFirstApply(_ExprTraversal):
    def handle_choice(self, choice):
        return (yield choice.chosen_outcome_or_default())

    def handle_expr(self, expr, **kwargs):
        if isinstance(expr._skrub_impl, Apply):
            raise _Found(expr)
        return (yield from super().handle_expr(expr, **kwargs))


def find_first_apply(expr):
    try:
        _FindFirstApply().run(expr)
    except _Found as first:
        return first.value
    return None


def supported_modes(expr):
    first = find_first_apply(expr)
    if first is None:
        return ["preview", "fit_transform", "transform"]
    return first._skrub_impl.supported_modes()
