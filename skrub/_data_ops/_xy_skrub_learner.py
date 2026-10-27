# Adapter to expose the standard scikit-learn interface fit(X, y) for a
# SkrubLearner (or ParamSearch or OptunaParamSearch).
# See _DataOpWrapperMixin.with_sklearn_interface for more information.
#
# Note: this may seem superficially similar to the _XyPipeline (and param
# search counterparts) defined in _estimator. However they serve different
# purposes and have different behavior.
#
# _XyPipeline is an internal class that is used in a transient way during
# parameter search and cross-validation. It stores the environment as an
# attribute and sends the X and y passed to estimator methods to the nodes
# marked with .skb.mark_as_X() and .skb.mark_as_y().
#
# The class defined here is for skrub users. It does _not_ store any data in
# attributes, instead other variables than X and y must be passed as kwargs
# (and the typical intended usage is for DataOps that do not require anything
# else than X and y). X and y are sent to the nodes with matching names or IDs.

import itertools
import re
from collections.abc import Mapping

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .._utils import set_module, short_repr
from . import _estimator, _evaluation, _optuna, _utils


def check_sklearn_interface(data_op, X, y):
    if not isinstance(X, (str, int)) or isinstance(X, bool):
        raise TypeError(
            "Expected X to be a string or int; "
            f"got object of type: {X.__class__.__name__!r}"
        )
    if y is not None and (not isinstance(y, (str, int)) or isinstance(y, bool)):
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


@set_module("skrub")
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

    if hasattr(BaseEstimator, "__sklearn_tags__"):

        def __sklearn_tags__(self):
            return self.learner.__sklearn_tags__()

    def __sklearn_is_fitted__(self):
        try:
            check_is_fitted(self.learner)
        except NotFittedError:
            return False
        return True

    def __getattr__(self, name):
        if name == "learner":
            # not set yet (e.g. during unpickling / copy): avoid infinite recursion
            _utils.attribute_error(self, name)
        if name.startswith("_") and name != "_estimator_type":
            _utils.attribute_error(self, name)
        if name not in ["preview", *_utils.ESTIMATOR_METHODS]:
            # not an estimator method: no (X, y) translation needed, and no need
            # to inspect the DataOp
            try:
                attr = getattr(self.learner, name)
                if name == "best_learner_":
                    attr = XySkrubLearner(attr, self.X, self.y)
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

    def get_param_grid(self, key_prefix="", outer_grid=None):
        """
        Get a param grid for this learner (use with GridSearchCV, RandomizedSearchCV).

        .. note::

            The recommended way to tune hyperparameters is to use the skrub
            DataOp methods :meth:`DataOp.skb.make_randomized_search()`,
            :meth:`DataOp.skb.make_grid_search()` rather than the scikit-learn
            meta-estimators.


        This method is meant to be used when the learner is a plain
        :class:`SkrubLearner`, i.e. it was created with
        :meth:`DataOp.skb.make_learner`. When the learner is itself a skrub
        hyperparameter search (e.g. created with
        :meth:`DataOp.skb.make_randomized_search`), it does its own internal
        tuning and from an external point of view it has no hyperparameters to
        tune, in this case ``outer_grid`` is returned unchanged.

        Parameters
        ----------
        key_prefix : str, default = ""
            Keys in the grid are prefixed with this string. Typically this is
            the path to reach the learner in a metaestimator such as a
            pipeline, for example ``'vectorizer__'`` if the learner is in a
            pipeline step named ``'vectorizer'``.

        outer_grid : dict, list of dicts or None, default = None
            Outer param grid for other steps in a pipeline. This function
            returns an augmented parameter grid in which the learner's
            parameter ranges have been inserted.

        Returns
        -------
        list of dict
            The parameter grid.

        Examples
        --------
        >>> import skrub
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.linear_model import Ridge

        >>> encoder = (
        ...     skrub.var("X")
        ...     .skb.apply(
        ...         skrub.StringEncoder(
        ...             n_components=skrub.choose_from([10, 20, 30], name="n_components")
        ...         )
        ...     )
        ...     .skb.make_learner()
        ...     .with_sklearn_interface()
        ... )

        >>> pipe = Pipeline([("encoder", encoder), ("ridge", Ridge())])
        >>> outer_grid = {"ridge__alpha": [.1, 1., 10.]}

        We obtain a grid that can be passed to GridSearchCV that can be used for tuning
        jointly the parameters of the Ridge and of the skrub learner:

        >>> encoder.get_param_grid("encoder__", outer_grid)
        [{'ridge__alpha': [0.1, 1.0, 10.0], 'encoder__0:n_components': ['0:10', '1:20', '2:30']}]
        """  # noqa: E501
        if not outer_grid:
            outer_grid = {}
        if isinstance(outer_grid, Mapping):
            outer_grid = [outer_grid]
        if isinstance(
            self.learner, (_estimator.ParamSearch, _optuna.OptunaParamSearch)
        ):
            # the search object has no params to tune in the outer search (it
            # does its own internal tuning).
            return list(outer_grid)
        graph = _evaluation.choice_graph(self.learner.data_op)
        learner_grid = []
        for subgrid in _evaluation.param_grid(self.learner.data_op):
            learner_subgrid = {}
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
                learner_subgrid[f"{key_prefix}{new_key}"] = new_value
            learner_grid.append(learner_subgrid)
        # As the outer grid keys are for other steps there should be no name
        # clash (+ the learner's key formatting is very specific). For simplicity
        # we don't do validation on the outer grid.
        return [
            out_g | learner_g
            for out_g, learner_g in itertools.product(outer_grid, learner_grid)
        ]
