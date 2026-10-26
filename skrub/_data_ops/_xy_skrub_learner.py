import re

from sklearn.base import BaseEstimator

from .._utils import short_repr
from . import _estimator, _evaluation, _utils


def check_sklearn_interface(data_op, X, y):
    if not isinstance(X, (str, int)):
        raise TypeError(
            "Expected X to be a string or int; "
            f"got object of type: {X.__class__.__name__!r}"
        )
    if y is not None and not isinstance(X, (str, int)):
        raise TypeError(
            "Expected y to be a string, int, or None; "
            f"got object of type: {y.__class__.__name__!r}"
        )
    all_nodes = _evaluation.nodes(data_op)
    all_names, all_ids = set(), set()
    for node in all_nodes:
        all_ids.add(node._skrub_impl.uuid)
        if (name := node._skrub_impl.name) is not None:
            all_names.add(name)
    if isinstance(X, str) and X not in all_names:
        raise ValueError(
            f"`with_sklearn_interface(X={X!r})` requires the DataOp to have "
            f"a variable or node named {X!r} but none was found."
        )
    if isinstance(X, int) and X not in all_ids:
        raise ValueError(
            f"`with_sklearn_interface(X={X!r})` requires the DataOp to have "
            f"a variable or node with ID {X!r} but none was found."
        )
    if y is None or not _estimator.data_op_needs_y(data_op):
        return
    if isinstance(y, str) and y not in all_names:
        raise ValueError(
            f"`with_sklearn_interface(y={y!r})` requires the DataOp to have "
            f"a variable or node named {y!r} but none was found."
        )
    if isinstance(y, int) and y not in all_ids:
        raise ValueError(
            f"`with_sklearn_interface(y={y!r})` requires the DataOp to have "
            f"a variable or node with ID {y!r} but none was found."
        )


class XySkrubLearner(BaseEstimator):
    """
    Wrapper to expose a standard scikit-learn interface to a skrub learner.

    This class is not usually instantiated manually, but by calling
    :meth:`.with_sklearn_interface() <SkrubLearner.with_sklearn_interface>`
    on a skrub learner. Skrub learners are created from a :class:`DataOp` with
    :meth:`.skb.make_learner() <DataOp.skb.make_learner>`,
    :meth:`.skb.make_randomized_search() <DataOp.skb.make_randomized_search>`,
    etc.

    The wrapped learner can be retrieved from the ``learner`` attribute.

    See :meth:`.with_sklearn_interface() <SkrubLearner.with_sklearn_interface>`
    for details.
    """

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
                    attr = XySkrubLearner(attr)
                return attr
            except AttributeError:
                _utils.attribute_error(self, name)
        if name not in _evaluation.supported_modes(self.learner.data_op):
            _utils.attribute_error(self, name)

        def f(X, y=None, **kwargs):
            env = kwargs | {self.X: X}
            if y is not None and self.y is not None:
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
