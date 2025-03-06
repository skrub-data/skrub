import inspect
from functools import partial

from sklearn import model_selection
from sklearn.base import BaseEstimator, clone

from .. import _join_utils
from .._parallel_plot import DEFAULT_COLORSCALE, plot_parallel_coord
from ._choosing import Choice, unwrap, unwrap_default
from ._evaluation import (
    choices,
    evaluate,
    find_node_by_name,
    find_X,
    find_y,
    get_params,
    needs_eval,
    nodes,
    param_grid,
    reachable,
    set_params,
)
from ._expressions import Apply, Expr
from ._utils import X_NAME, Y_NAME, attribute_error


def _prune_cache(expr, mode, *args, **kwargs):
    reachable_nodes = reachable(expr, mode)
    for node in nodes(expr):
        if id(node) not in reachable_nodes:
            node._skrub_impl.results.pop(mode, None)


def _check_env(environment, caller_name):
    """Helper to detect the mistake eg fit(X) instead of fit({'X': X})"""
    if not isinstance(environment, dict):
        raise TypeError(
            f"The first argument to {caller_name!r} should be a dictionary of input"
            f" values, for example: {caller_name}({{'X': df, 'other_table_name':"
            " other_df, ...})"
        )
    env_contains_expr, found_node = needs_eval(environment, return_node=True)
    if env_contains_expr:
        if isinstance(found_node, Expr):
            description = f"a skrub expression: {found_node._skrub_impl!r}"
        else:
            description = f"a skrub choice: {found_node}"
        raise TypeError(
            f"The `environment` dict passed to {caller_name!r} "
            f"contains {description}. This argument should only "
            "contain actual values on which to run the computation."
        )
    # TODO: here we could check that there are no extra keys in `environment`,
    #       ie all keys in `environment` correspond to a name in the expression.
    #
    # Note: we cannot check that all variables in the expression have a
    # matching key in the `environment`, because depending on the mode,
    # choices, and result of dynamic conditional expressions such as
    # `.skb.if_else()` some variables in the expression may not be required for
    # evaluation and those do not need a value.


class ExprEstimator(BaseEstimator):
    def __init__(self, expr):
        self.expr = expr

    def __skrub_to_sklearn_compatible__(self, environment):
        return CompatibleExprEstimator(self.expr.skb.clone(), environment)

    def fit(self, environment):
        _check_env(environment, "fit")
        _ = self.fit_transform(environment)
        return self

    def fit_transform(self, environment):
        # TODO: not needed, can be handled by _eval_in_mode?
        _check_env(environment, "fit_transform")
        callback = partial(_prune_cache, self.expr, "fit_transform")
        env = environment | {"_callback": callback}
        return evaluate(self.expr, "fit_transform", env, clear=True)

    def _eval_in_mode(self, mode, environment):
        _check_env(environment, mode)
        callback = partial(_prune_cache, self.expr, mode)
        env = environment | {"_callback": callback}
        return evaluate(self.expr, mode, env, clear=True)

    def report(self, mode, environment, **full_report_kwargs):
        from ._inspection import full_report

        full_report_kwargs["clear"] = True
        return full_report(
            self.expr, environment=environment, mode=mode, **full_report_kwargs
        )

    def __getattr__(self, name):
        if name not in self.expr._skrub_impl.supports_modes():
            attribute_error(self, name)

        def f(*args, **kwargs):
            return self._eval_in_mode(name, *args, **kwargs)

        f.__name__ = name
        return f

    def get_params(self, deep=True):
        params = {"expr": self.expr}
        if not deep:
            return params
        params.update({f"expr__{k}": v for k, v in get_params(self.expr).items()})
        return params

    def set_params(self, **params):
        params = {k: unwrap(v) for k, v in params.items()}
        if "expr" in params:
            self.expr = params.pop("expr")
        set_params(self.expr, {int(k.lstrip("expr__")): v for k, v in params.items()})
        return self

    def sub_estimator(self, name):
        node = find_node_by_name(self.expr, name)
        if node is None:
            return None
        impl = node._skrub_impl
        if not isinstance(impl, Apply):
            raise TypeError(
                f"node {name!r} does not represent a sub-estimator: {node!r}"
            )
        if not hasattr(impl, "estimator_"):
            raise ValueError(
                f"Node {name!r} has not been fitted. Call fit() on the estimator "
                "before attempting to retrieve fitted sub-estimators."
            )
        return node._skrub_impl.estimator_


class _SklearnCompatibleMixin:
    def __sklearn_clone__(self):
        params = list(inspect.signature(self.__init__).parameters)
        kwargs = {p: clone(getattr(self, p)) for p in params if p != "environment"}
        return self.__class__(**kwargs, environment=self.environment)

    def __skrub_clear_environment__(self):
        self.environment = None

    def _pick_env(self, environment):
        assert (self.environment is None) ^ (environment is None)
        return self.environment if environment is None else environment


def _to_sklearn_compatible(estimator, environment):
    try:
        return estimator.__skrub_to_sklearn_compatible__(environment)
    except AttributeError:
        return clone(estimator)


def _clear_environment(estimator):
    try:
        estimator.__skrub_clear_environment__()
    except AttributeError:
        pass


class CompatibleExprEstimator(_SklearnCompatibleMixin, ExprEstimator):
    def __init__(self, expr, environment):
        self.expr = expr
        self.environment = environment

    @property
    def _estimator_type(self):
        try:
            return unwrap_default(self.expr._skrub_impl.estimator)._estimator_type
        except AttributeError:
            return "transformer"

    @property
    def __sklearn_tags__(self):
        try:
            return unwrap_default(self.expr._skrub_impl.estimator).__sklearn_tags__
        except AttributeError:
            attribute_error(self, "__sklearn_tags__")

    @property
    def classes_(self):
        try:
            estimator = self.expr._skrub_impl.estimator_
        except AttributeError:
            attribute_error(self, "classes_")
        return estimator.classes_

    def fit_transform(self, X, y=None, environment=None):
        environment = self._pick_env(environment)
        callback = partial(_prune_cache, self.expr, "fit_transform")
        xy_environment = {X_NAME: X, "_callback": callback}
        if y is not None:
            xy_environment[Y_NAME] = y
        xy_environment = {**environment, **xy_environment}
        return evaluate(self.expr, "fit_transform", xy_environment, clear=True)

    def fit(self, X, y=None, environment=None):
        _ = self.fit_transform(X, y=y, environment=environment)
        return self

    def _eval_in_mode(self, mode, X, y=None, environment=None):
        environment = self._pick_env(environment)
        callback = partial(_prune_cache, self.expr, mode)
        xy_environment = {X_NAME: X, "_callback": callback}
        if y is not None:
            xy_environment[Y_NAME] = y
        xy_environment = {**environment, **xy_environment}
        return evaluate(self.expr, mode, xy_environment, clear=True)


def cross_validate(expr_estimator, environment, scoring=None, **cv_params):
    expr = expr_estimator.expr
    X_y = _find_X_y(expr)
    X = evaluate(X_y["X"].skb.clone(), "fit_transform", environment)
    if "y" in X_y:
        y = evaluate(X_y["y"].skb.clone(), "fit_transform", environment)
    else:
        y = None

    estimator = _to_sklearn_compatible(expr_estimator, environment)

    result = model_selection.cross_validate(
        estimator,
        X,
        y,
        scoring=scoring,
        **cv_params,
    )
    if (estimators := result.get("estimator", None)) is None:
        return result
    for e in estimators:
        _clear_environment(e)
    return result


def _find_X_y(expr):
    x_node = find_X(expr)
    if x_node is None:
        raise ValueError('expr should have a node marked with "mark_as_X()"')
    result = {"X": x_node}
    if (y_node := find_y(expr)) is not None:
        result["y"] = y_node
    else:
        impl = expr._skrub_impl
        if getattr(impl, "y", None) is not None:
            # the final estimator requests a y so some node must have been
            # marked as y
            raise ValueError('expr should have a node marked with "mark_as_y()"')
    return result


# TODO with ParameterGrid and ParameterSampler we can generate the list of
# candidates so we can provide more than just a score, eg full predictions for
# each sampled param combination.


class ParamSearch(BaseEstimator):
    def __init__(self, expr, search):
        self.expr = expr
        self.search = search

    def __skrub_to_sklearn_compatible__(self, environment):
        return CompatibleParamSearch(
            self.expr.skb.clone(), clone(self.search), environment
        )

    def fit(self, environment):
        X_y = _find_X_y(self.expr)
        X = evaluate(X_y["X"].skb.clone(), "fit_transform", environment)
        if "y" in X_y:
            y = evaluate(X_y["y"].skb.clone(), "fit_transform", environment)
        else:
            y = None
        self.estimator_ = CompatibleExprEstimator(self.expr.skb.clone(), environment)
        self.search_ = clone(self.search)
        self.search_.estimator = self.estimator_
        param_grid = self._get_param_grid()
        if hasattr(self.search_, "param_grid"):
            self.search_.param_grid = param_grid
        else:
            assert hasattr(self.search_, "param_distributions")
            self.search_.param_distributions = param_grid
        try:
            self.search_.fit(X, y)
        finally:
            # TODO copy useful attributes and stop storing self.search_ instead
            _clear_environment(self.estimator_)
            _clear_environment(self.search_.best_estimator_)
        return self

    def _get_param_grid(self):
        grid = param_grid(self.estimator_.expr)
        new_grid = []
        for subgrid in grid:
            subgrid = {f"expr__{k}": v for k, v in subgrid.items()}
            new_grid.append(subgrid)
        return new_grid

    def __getattr__(self, name):
        if name == "search_":
            attribute_error(self, name)
        if name not in self.expr._skrub_impl.supports_modes():
            return getattr(self.search_, name)

        def f(*args, **kwargs):
            return self._call_predictor_method(name, *args, **kwargs)

        f.__name__ = name
        return f

    def _call_predictor_method(self, name, environment):
        if not hasattr(self, "search_"):
            raise ValueError("Search not fitted")
        return getattr(self.best_estimator_, name)(environment)

    @property
    def best_estimator_(self):
        if not hasattr(self, "search_"):
            attribute_error(self, "best_estimator_")
        return ExprEstimator(self.search_.best_estimator_.expr)

    @property
    def results_(self):
        return self._get_cv_results_table()

    def _get_cv_results_table(self, return_metadata=False, detailed=False):
        import pandas as pd

        expr_choices = choices(self.estimator_.expr)

        all_rows = []
        param_names = set()
        log_scale_columns = set()
        for params in self.cv_results_["params"]:
            row = {}
            for param_id, param in params.items():
                choice = expr_choices[int(param_id.lstrip("expr__"))]
                if isinstance(choice, Choice):
                    param = choice.outcomes[param]
                choice_name = param.in_choice or param_id
                value = param.name or param.value
                row[choice_name] = value
                param_names.add(choice_name)
                if getattr(param, "is_from_log_scale", False):
                    log_scale_columns.add(choice_name)
            all_rows.append(row)

        metadata = {"log_scale_columns": list(log_scale_columns)}
        # all_ordered_param_names = _get_all_param_names(self._get_param_grid())
        # ordered_param_names = [n for n in all_ordered_param_names if n in param_names]
        # table = pd.DataFrame(all_rows, columns=ordered_param_names)
        table = pd.DataFrame(all_rows)
        result_keys = [
            "mean_test_score",
            "std_test_score",
            "mean_fit_time",
            "std_fit_time",
            "mean_score_time",
            "std_score_time",
            "mean_train_score",
            "std_train_score",
        ]
        new_names = _join_utils.pick_column_names(table.columns, result_keys)
        renaming = dict(zip(table.columns, new_names))
        table.columns = new_names
        metadata["log_scale_columns"] = [
            renaming[c] for c in metadata["log_scale_columns"]
        ]
        table.insert(0, "mean_test_score", self.cv_results_["mean_test_score"])
        if detailed:
            for k in result_keys[1:]:
                if k in self.cv_results_:
                    table.insert(table.shape[1], k, self.cv_results_[k])
        table = table.sort_values("mean_test_score", ascending=False, ignore_index=True)
        return (table, metadata) if return_metadata else table

    def plot_results(self, *, colorscale=DEFAULT_COLORSCALE, min_score=None):
        cv_results, metadata = self._get_cv_results_table(
            return_metadata=True, detailed=True
        )
        cv_results = cv_results.drop(
            [
                "std_test_score",
                "std_fit_time",
                "std_score_time",
                "mean_train_score",
                "std_train_score",
            ],
            axis="columns",
            errors="ignore",
        )
        if min_score is not None:
            cv_results = cv_results[cv_results["mean_test_score"] >= min_score]
        return plot_parallel_coord(cv_results, metadata, colorscale=colorscale)


def _get_all_param_names(grid):
    names = {}
    for subgrid in grid:
        for k, v in subgrid.items():
            if v.name is not None:
                k = v.name
            names[k] = None
    return list(names)


class CompatibleParamSearch(_SklearnCompatibleMixin, ParamSearch):
    def __init__(self, expr, search, environment):
        self.expr = expr
        self.search = search
        self.environment = environment

    @property
    def _estimator_type(self):
        try:
            return unwrap_default(self.expr._skrub_impl.estimator)._estimator_type
        except AttributeError:
            return "transformer"

    @property
    def classes_(self):
        try:
            estimator = self.best_estimator_.expr._skrub_impl.estimator_
        except AttributeError:
            attribute_error(self, "classes_")
        return estimator.classes_

    def fit(self, X, y=None, environment=None):
        environment = self._pick_env(environment)
        xy_environment = {X_NAME: X}
        if y is not None:
            xy_environment[Y_NAME] = y
        xy_environment = {**environment, **xy_environment}
        super().fit(xy_environment)
        return self

    def _call_predictor_method(self, name, X, y=None, environment=None):
        environment = self._pick_env(environment)
        if not hasattr(self, "search_"):
            raise ValueError("Search not fitted")
        return getattr(self.search_.best_estimator_, name)(
            X, y=y, environment=environment
        )
