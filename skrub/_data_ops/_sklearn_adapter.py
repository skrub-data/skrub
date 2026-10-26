import re

from sklearn.base import BaseEstimator

from .._utils import short_repr
from . import _estimator, _evaluation, _utils


def check_sklearn_interface(data_op, X, y):
    all_names = data_op.skb.get_vars(all_named_ops=True)
    if X not in all_names:
        raise ValueError(
            f"`with_sklearn_interface(X={X!r})` requires the DataOp to have "
            f"a variable or node named {X!r} but none was found."
        )
    if not _estimator.data_op_needs_y(data_op):
        return
    if y not in all_names:
        raise ValueError(
            f"`with_sklearn_interface(y={y!r})` requires the DataOp to have "
            f"a variable or node named {y!r} but none was found."
        )


class SklearnAdapter(BaseEstimator):
    def __init__(self, learner, X="X", y="y"):
        self.learner = learner
        self.X = X
        self.y = y

    def set_params(self, **params):
        dataop_params, rest = {}, {}
        for k, v in params.items():
            if re.match(r"\d+:", k):
                dataop_params[k] = v
            else:
                rest[k] = v
        super().set_params(**rest)
        self.learner.set_params(**dataop_params)
        return self

    def get_named_params(self):
        return self.learner.get_named_params()

    def set_named_params(self, **params):
        self.learner.set_named_params(**params)
        return self

    def __sklearn_tags__(self):
        return self.learner.__sklearn_tags__()

    def __getattr__(self, name):
        if name == "learner":
            # not set yet (e.g. during unpickling / copy): avoid infinite recursion
            _utils.attribute_error(self, name)
        if name not in ["preview", *_utils.ESTIMATOR_METHODS]:
            # not an estimator method: no (X, y) translation needed, and no need
            # to inspect the DataOp
            try:
                attr = getattr(self.learner, name)
                if name == "best_learner_":
                    attr = SklearnAdapter(attr)
                return attr
            except AttributeError:
                _utils.attribute_error(self, name)
        if name not in _evaluation.supported_modes(self.learner.data_op):
            _utils.attribute_error(self, name)

        def f(X, y=None, **kwargs):
            env = kwargs | {self.X: X}
            if y is not None:
                env[self.y] = y
            result = getattr(self.learner, name)(env)
            # ``fit`` must return the adapter to allow ``est.fit(X, y).predict(X)``
            return self if name == "fit" else result

        f.__name__ = name
        return f

    def get_param_grid(self, key_prefix=""):
        param_grid = _evaluation.param_grid(self.learner.data_op)
        graph = _evaluation.choice_graph(self.learner.data_op)
        new_grid = []
        for subgrid in param_grid:
            new_subgrid = {}
            for choice_id, choice_range in subgrid.items():
                choice = graph["choices"][choice_id]
                new_key = f"{choice_id}:{graph['choice_display_names'][choice_id]}"
                if hasattr(choice, "outcomes"):
                    if choice.outcome_names is None:
                        outcome_names = list(map(short_repr, choice.outcomes))
                    else:
                        outcome_names = choice.outcome_names
                    new_value = [
                        f"{outcome_idx}:{outcome_names[outcome_idx]}"
                        for outcome_idx in choice_range
                    ]
                else:
                    new_value = choice_range
                new_subgrid[f"{key_prefix}{new_key}"] = new_value
            new_grid.append(new_subgrid)
        return new_grid
