from sklearn.base import BaseEstimator, clone

from .._utils import short_repr
from . import _estimator, _evaluation, _utils


def can_use_sklearn_interface(data_op):
    return "X" in data_op.skb.get_vars(all_named_ops=True)


class SklearnAdapter(BaseEstimator):
    def __init__(self, learner):
        self.learner = learner

    def get_params(self):
        return self.learner.get_params()

    def set_params(self, **params):
        self.learner.set_params(**params)
        return self

    def __sklearn_clone__(self):
        return self.__class__(clone(self.learner))

    def __sklearn_tags__(self):
        return self.learner.__sklearn_tags__()

    def _get_params_html(self, deep=True, doc_link=""):
        # Function called by scikit-learn to display the estimator params in
        # diagram.
        # We override it to display the graph instead of just the default which
        # would be 'data_op: repr(self.data_op)'

        data_op_html = self.learner.data_op._repr_html_()
        sklearn_params = super()._get_params_html()
        sklearn_params.pop("data_op")
        if sklearn_params:
            sklearn_html = sklearn_params._repr_html_inner()
            html = f"<div><div>{sklearn_html}</div><div>{data_op_html}</div></div>"
        else:
            html = f"<div>{data_op_html}</div>"
        return _estimator._SklearnParamsReprHtml(html)

    def __getattr__(self, name):
        if name == "learner":
            # not set yet (e.g. during unpickling / copy): avoid infinite recursion
            _utils.attribute_error(self, name)
        if name not in _evaluation.supported_modes(self.learner.data_op):
            try:
                return getattr(self.learner, name)
            except AttributeError:
                _utils.attribute_error(self, name)

        def f(X, y=None, **kwargs):
            env = kwargs | {"X": X}
            if y is not None:
                env["y"] = y
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
                    outcome_names = [f"{i}:{n}" for i, n in enumerate(outcome_names)]
                    new_value = [
                        f"{outcome_idx}:{outcome_names[outcome_idx]}"
                        for outcome_idx in choice_range
                    ]
                else:
                    new_value = choice_range
                new_subgrid[f"{key_prefix}{new_key}"] = new_value
            new_grid.append(new_subgrid)
        return new_grid


class XySkrubLearner(SklearnAdapter):
    pass


class XyParamSearch(SklearnAdapter):
    pass


class XyOptunaSearch(SklearnAdapter):
    pass
