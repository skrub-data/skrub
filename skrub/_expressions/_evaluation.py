import copy
import functools
import inspect
import types
import warnings
from collections import defaultdict
from types import SimpleNamespace

from sklearn.base import BaseEstimator
from sklearn.base import clone as skl_clone

from . import _choosing
from ._expressions import (
    _BUILTIN_MAP,
    _BUILTIN_SEQ,
    Expr,
    IfElse,
    Value,
    Var,
    _Constants,
)
from ._utils import X_NAME, Y_NAME, simple_repr

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
    "reachable",
    "get_params",
    "set_params",
]


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
                    if isinstance(top._skrub_impl, IfElse):
                        push(self.handle_if_else)
                    else:
                        push(self.handle_expr)
                elif isinstance(top, _BUILTIN_MAP):
                    push(self.handle_mapping)
                elif isinstance(top, _BUILTIN_SEQ):
                    push(self.handle_seq)
                elif isinstance(top, slice):
                    push(self.handle_slice)
                elif isinstance(top, _choosing.BaseChoice):
                    push(self.handle_choice)
                elif isinstance(top, _choosing.Outcome):
                    push(self.handle_outcome)
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

    def handle_if_else(self, expr):
        return (yield from self.handle_expr(expr))

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

    def handle_outcome(self, outcome):
        value = yield outcome.value
        return _choosing._with_fields(outcome, value=value)

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
        new_seq = []
        for item in seq:
            value = yield item
            new_seq.append(value)
        return type(seq)(new_seq)

    def handle_mapping(self, mapping):
        new_mapping = {}
        for k, v in mapping.items():
            new_mapping[(yield k)] = yield v
        return type(mapping)(new_mapping)

    def handle_slice(self, s):
        return slice((yield s.start), (yield s.stop), (yield s.step))


class _Evaluator(_ExprTraversal):
    def __init__(self, mode="preview", environment=None, callback=None):
        self.mode = mode
        self.environment = {} if environment is None else environment
        self.callback = callback

    def _pick_mode(self, expr):
        if expr is not self._expr and self.mode != "preview":
            return "fit_transform" if "fit" in self.mode else "transform"
        return self.mode

    def run(self, expr):
        self._expr = expr
        return super().run(expr)

    def handle_if_else(self, expr):
        cond = yield expr._skrub_impl.condition
        if cond:
            return (yield expr._skrub_impl.value_if_true)
        else:
            return (yield expr._skrub_impl.value_if_false)

    def handle_expr(self, expr):
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
        if self.mode in impl.results:
            return impl.results[self.mode]
        result = yield from super().handle_expr(
            expr, impl.fields_required_for_eval(self._pick_mode(expr))
        )
        impl.results[self.mode] = result
        if self.callback is not None:
            self.callback(expr, result)
        return result

    def handle_choice(self, choice):
        if choice.name is not None and choice.name in self.environment:
            return self.environment[choice.name]
        if self.mode == "preview":
            return (yield _choosing.unwrap_default(choice))
        outcome = choice.chosen_outcome_or_default()
        return (yield outcome)

    def handle_outcome(self, outcome):
        return (yield _choosing.unwrap(outcome))

    def handle_choice_match(self, choice_match):
        outcome = yield choice_match.choice
        return (yield choice_match.outcome_mapping[outcome])

    def compute_result(self, expr, evaluated_attributes):
        mode = self._pick_mode(expr)
        try:
            return expr._skrub_impl.compute(
                SimpleNamespace(**evaluated_attributes),
                mode=mode,
                environment=self.environment,
            )
        except Exception as e:
            expr._skrub_impl.errors[mode] = e
            if mode == "preview":
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


def evaluate(expr, mode="preview", environment=None, callback=None, clear=False):
    if clear:
        clear_results(expr, mode=mode)
    try:
        return _Evaluator(mode=mode, environment=environment, callback=callback).run(
            expr
        )
    finally:
        if clear:
            clear_results(expr, mode=mode)


class _Reachable(_Evaluator):
    def __init__(self, mode):
        self.mode = mode
        self.callback = None

    def run(self, expr):
        self._reachable = {}
        super().run(expr)
        return self._reachable

    def handle_expr(self, expr):
        self._reachable[id(expr)] = expr
        return (yield from super().handle_expr(expr))


def reachable(expr, mode):
    return _Reachable(mode).run(expr)


class _Printer(_ExprTraversal):
    def __init__(self, highlight=False):
        self.highlight = highlight

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
        open_tag, close_tag = "", ""
        if not is_seen and self.highlight:
            open_tag, close_tag = "\033[1m", "\033[0m"
        line = simple_repr(expr, open_tag, close_tag)
        if is_seen:
            line = f"( {line} )*"
            self._cache_used = True
        self._lines.append(line)
        self._seen.add(id(expr))


def describe_steps(expr, highlight=False):
    return _Printer(highlight=highlight).run(expr)


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
        clone_impl = impl.__class__(**evaluated_attributes)
        if isinstance(clone_impl, Var) and self.drop_preview_data:
            clone_impl.value = _Constants.NO_VALUE
        clone_impl.is_X = impl.is_X
        clone_impl.is_y = impl.is_y
        clone_impl._creation_stack_lines = impl._creation_stack_lines
        clone_impl.name = impl.name
        clone = Expr(clone_impl)
        self._replace[id(expr)] = clone
        return clone


def clone(expr, replace=None, drop_preview_data=False):
    return _Cloner(replace=replace, drop_preview_data=drop_preview_data).run(expr)


def _unique(seq):
    return list(dict.fromkeys(seq))


def _simplify_graph(graph):
    short = {v: i for i, v in enumerate(graph["nodes"].keys())}
    new_nodes = {short[k]: v for k, v in graph["nodes"].items()}
    new_parents = {
        short[k]: [short[p] for p in _unique(v)] for k, v in graph["parents"].items()
    }
    new_children = {
        short[k]: [short[c] for c in _unique(v)] for k, v in graph["children"].items()
    }
    return {"nodes": new_nodes, "parents": new_parents, "children": new_children}


class _Graph(_ExprTraversal):
    def run(self, expr):
        self._nodes = {}
        self._parents = defaultdict(list)
        self._children = defaultdict(list)
        self._current_expr = []
        _ = super().run(expr)
        graph = {
            "nodes": self._nodes,
            "parents": dict(self._parents),
            "children": dict(self._children),
        }
        return _simplify_graph(graph)

    def handle_expr(self, expr):
        if self._current_expr:
            parent, child = id(expr), id(self._current_expr[-1])
            self._parents[child].append(parent)
            self._children[parent].append(child)
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
        self._outcomes = {}
        self._parents = defaultdict(set)
        self._current_outcome = [None]
        _ = super().run(expr)
        short = {v: i for i, v in enumerate(self._choices.keys())}
        self._short_ids = short
        choices = {short[k]: v for k, v in self._choices.items()}
        parents = {k: [short[p] for p in v] for k, v in self._parents.items()}
        # - choices: choice's short id (1, 2, ...) to BaseChoice instance
        # - parents: outcome's id (id(outcome)) to list of its parent choices' short ids
        return {"choices": choices, "parents": parents}

    def handle_choice(self, choice):
        # unlike during evaluation here we need pre-ordering
        self._parents[self._current_outcome[-1]].add(id(choice))
        self._choices[id(choice)] = choice
        yield from super().handle_choice(choice)
        return choice

    def handle_outcome(self, outcome):
        self._current_outcome.append(id(outcome))
        yield from super().handle_outcome(outcome)
        self._current_outcome.pop()
        return outcome

    def handle_choice_match(self, choice_match):
        yield choice_match.choice
        for outcome in choice_match.choice.outcomes:
            value = outcome.value
            self._current_outcome.append(id(outcome))
            yield choice_match.outcome_mapping[value]
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
    x_y_choices = set()
    for node in [find_X(expr), find_y(expr)]:
        if node is not None:
            node_graph = _ChoiceGraph().run(node)
            x_y_choices.update(
                {
                    # convert from python id to short id
                    full_builder._short_ids[id(c)]
                    for c in node_graph["choices"].values()
                }
            )
    if x_y_choices:
        warnings.warn(
            "The following choices are used in the construction of X or y, "
            "so their value cannot be tuned because they are needed outside "
            "of the cross-validation loop. They will be clamped to their "
            f"default value: {[full_graph['choices'][k] for k in x_y_choices]}"
        )
    full_graph["x_y_choices"] = x_y_choices
    return full_graph


def _expand_grid(graph, grid):
    def choice_range(choice_id):
        # The range of possible values for a choice.
        # for numeric choices it is the object itself and for Choices the range
        # of possible outcome indices.
        # if the choice is used in X or y, it is clamped to a single value
        choice = graph["choices"][choice_id]
        if choice_id in graph["x_y_choices"]:
            if isinstance(choice, _choosing.Choice):
                return [choice.chosen_outcome_idx or 0]
            else:
                return [choice.default()]
        else:
            if isinstance(choice, _choosing.Choice):
                return list(range(len(choice.outcomes)))
            else:
                return choice

    def has_parents(choice_id):
        # if any of the outcomes in a choice contains another choice. in this
        # case it needs to be on a separate subgrid.
        choice = graph["choices"][choice_id]
        if not isinstance(choice, _choosing.Choice):
            return False
        for outcome in choice.outcomes:
            if graph["parents"].get(id(outcome), None):
                return True
        return False

    # extract
    if None not in graph["parents"]:
        return [grid]
    for choice_id in graph["parents"][None]:
        if not has_parents(choice_id):
            grid[choice_id] = choice_range(choice_id)
    # split
    remaining = [c_id for c_id in graph["parents"][None] if c_id not in grid]
    if not remaining:
        return [grid]
    choice_id = remaining[0]
    subgrids = []
    for outcome_idx in choice_range(choice_id):
        outcome = graph["choices"][choice_id].outcomes[outcome_idx]
        new_subgrid = grid.copy()
        graph = graph.copy()
        graph["parents"][None] = graph["parents"].get(id(outcome), []) + remaining[1:]
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
                "Only one node can be marked with `mark_as_x()`. "
                "2 different objects were marked as X:\n"
                f"first object that used `.mark_as_x()`:\n{first}\n"
                f"second object that used `.mark_as_x()`:\n{second}"
            )
        if conflict["reason"] == "is_y":
            return (
                "Only one node can be marked with `mark_as_y()`. "
                "2 different objects were marked as y:\n"
                f"first object that used `.mark_as_y()`:\n{first}\n"
                f"second object that used `.mark_as_y()`:\n{second}"
            )
        if conflict["reason"] == "name":
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
