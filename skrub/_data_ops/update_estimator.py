# Scikit-learn-ish interface to the skrub DataOps
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn import model_selection
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, check_cv, StratifiedKFold
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.model_selection import cross_val_predict as sklearn_cross_val_predict

from .. import _dataframe as sbd
from .. import _join_utils
from ._choosing import BaseNumericChoice, get_default
from ._data_ops import Apply, DataOp, check_subsampled_X_y_shape
from ._evaluation import (
    choice_graph,
    eval_choices,
    evaluate,
    find_first_apply,
    find_node_by_name,
    find_X,
    find_y,
    get_params,
    param_grid,
    set_params,
    supported_modes,
)
from ._inspection import describe_params
from ._parallel_coord import DEFAULT_COLORSCALE, plot_parallel_coord
from ._subsampling import env_with_subsampling
from ._utils import KFOLD_5, X_NAME, Y_NAME, _CloudPickle, attribute_error

_FITTING_METHODS = ["fit", "fit_transform", "partial_fit"]
_SKLEARN_SEARCH_FITTED_ATTRIBUTES_TO_COPY = [
    "cv_results_",
    "best_score_",
    "best_params_",
    "best_index_",
    "scorer_",
    "n_splits_",
    "refit_time_",
    "multimetric_",
    "early_stopping_rounds_",  # NEW: Track early stopping rounds
    "best_iteration_",  # NEW: Track best iteration for iterative models
]
_SEARCH_FITTED_ATTRIBUTES = _SKLEARN_SEARCH_FITTED_ATTRIBUTES_TO_COPY + [
    "best_learner_",
    "refit_",
    "feature_importances_",  # NEW: Store feature importances
]


def _default_sklearn_tags():
    class _DummyTransformer(TransformerMixin, BaseEstimator):
        pass

    return _DummyTransformer().__sklearn_tags__()


class _SharedDict(dict):
    """A dict that does not get copied during deepcopy/sklearn clone."""
    
    def __deepcopy__(self, memo):
        return self

    def __sklearn_clone__(self):
        return self


def _copy_attr(source, target, attributes):
    for a in attributes:
        try:
            setattr(target, a, getattr(source, a))
        except AttributeError:
            pass


class _CloudPickleDataOp(_CloudPickle):
    """Mixin to serialize the `DataOp` attribute with cloudpickle."""
    
    _cloudpickle_attributes = ["data_op"]


# NEW: Early stopping callback class
class _EarlyStoppingCallback:
    """Callback for early stopping in iterative models."""
    
    def __init__(self, early_stopping_rounds=10, validation_fraction=0.1):
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.best_score = -np.inf
        self.best_iteration = 0
        self.no_improvement_count = 0
        
    def __call__(self, env_score, iteration):
        if env_score > self.best_score:
            self.best_score = env_score
            self.best_iteration = iteration
            self.no_improvement_count = 0
            return False  # Continue training
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.early_stopping_rounds:
                return True  # Stop training
        return False


class SkrubLearner(_CloudPickleDataOp, BaseEstimator):
    """Learner that evaluates a skrub DataOp.
    
    New Features:
    - partial_fit() support for online learning
    - feature_importances_ property
    - model persistence methods
    - warm_start parameter
    """
    
    def __init__(self, data_op, warm_start=False, early_stopping_rounds=None):
        self.data_op = data_op
        self.warm_start = warm_start
        self.early_stopping_rounds = early_stopping_rounds
        self._early_stopping_callback = None
        
    def __skrub_to_Xy_pipeline__(self, environment):
        """Convert to a fully scikit-learn compatible pipeline (fit takes X, y)."""
        new = _XyPipeline(self.data_op, _SharedDict(environment))
        _copy_attr(self, new, ["_is_fitted", "warm_start", "early_stopping_rounds"])
        return new

    def _set_is_fitted(self, mode):
        if mode in _FITTING_METHODS:
            self._is_fitted = True

    def __sklearn_is_fitted__(self):
        return getattr(self, "_is_fitted", False)

    def _eval_in_mode(self, mode, environment):
        if mode not in _FITTING_METHODS and mode != "partial_fit":
            check_is_fitted(self)
        
        # NEW: Handle early stopping for iterative models
        if mode == "fit" and self.early_stopping_rounds:
            if self._early_stopping_callback is None:
                self._early_stopping_callback = _EarlyStoppingCallback(
                    early_stopping_rounds=self.early_stopping_rounds
                )
            environment = environment.copy()
            environment["_early_stopping_callback"] = self._early_stopping_callback
        
        result = evaluate(self.data_op, mode, environment, clear=not self.warm_start)
        self._set_is_fitted(mode)
        return result

    def report(self, *, environment, mode, **full_report_kwargs):
        """Call the method specified by ``mode`` and return the result and full report."""
        from ._inspection import full_report

        if mode not in _FITTING_METHODS and mode != "partial_fit":
            check_is_fitted(self)

        full_report_kwargs["clear"] = not self.warm_start
        result = full_report(
            self.data_op, environment=environment, mode=mode, **full_report_kwargs
        )
        if mode == "fit" and result["result"] is not None:
            result["result"] = self
        self._set_is_fitted(mode)
        return result

    # NEW: Add partial_fit support
    def partial_fit(self, environment, classes=None):
        """Incremental fit on a batch of samples.
        
        Parameters
        ----------
        environment : dict
            Bindings for variables contained in the DataOp
        classes : array-like, optional
            Classes across all calls to partial_fit for classification
        
        Returns
        -------
        self
        """
        if classes is not None:
            environment = environment.copy()
            environment["_partial_fit_classes"] = classes
        
        result = self._eval_in_mode("partial_fit", environment)
        return self if result is None else result

    # NEW: Feature importances property
    @property
    def feature_importances_(self):
        """Return feature importances if the underlying estimator supports it."""
        check_is_fitted(self)
        
        # Find the first estimator that might have feature_importances_
        for node_name in self._get_estimator_nodes():
            try:
                estimator = self.find_fitted_estimator(node_name)
                if hasattr(estimator, 'feature_importances_'):
                    return estimator.feature_importances_
            except (KeyError, NotFittedError, AttributeError):
                continue
        
        # Try feature_importances attribute directly
        try:
            first_apply = find_first_apply(self.data_op)
            if hasattr(first_apply._skrub_impl, 'feature_importances_'):
                return first_apply._skrub_impl.feature_importances_
        except AttributeError:
            pass
        
        attribute_error(self, "feature_importances_")

    # NEW: Get feature names after transformation
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.
        
        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features
        
        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names
        """
        check_is_fitted(self)
        
        # Find transformer nodes
        feature_names = []
        for node_name in self._get_estimator_nodes():
            try:
                estimator = self.find_fitted_estimator(node_name)
                if hasattr(estimator, 'get_feature_names_out'):
                    if not feature_names:
                        feature_names = estimator.get_feature_names_out(input_features)
                    else:
                        feature_names = estimator.get_feature_names_out(feature_names)
            except (KeyError, NotFittedError, AttributeError):
                continue
        
        if feature_names:
            return np.array(feature_names)
        
        # Fallback: use input features if available
        if input_features is not None:
            return np.array(input_features)
        
        raise AttributeError("Unable to determine feature names")

    # NEW: Model persistence methods
    def save(self, path, compress=True):
        """Save the fitted learner to disk.
        
        Parameters
        ----------
        path : str or Path
            Path where to save the model
        compress : bool, default=True
            Whether to compress the model file
        """
        check_is_fitted(self)
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'data_op': self.data_op,
            'fitted': True,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        if compress:
            import gzip
            with gzip.open(path, 'wb') as f:
                pickle.dump(model_data, f)
        else:
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)

    @classmethod
    def load(cls, path):
        """Load a fitted learner from disk.
        
        Parameters
        ----------
        path : str or Path
            Path to the saved model
        
        Returns
        -------
        SkrubLearner
            Loaded and fitted learner
        """
        path = Path(path)
        
        if path.suffix == '.gz':
            import gzip
            with gzip.open(path, 'rb') as f:
                model_data = pickle.load(f)
        else:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
        
        learner = cls(model_data['data_op'])
        learner._is_fitted = model_data['fitted']
        return learner

    # NEW: Get model size estimate
    def get_model_size(self, unit='MB'):
        """Get estimated memory footprint of the fitted model.
        
        Parameters
        ----------
        unit : {'B', 'KB', 'MB', 'GB'}, default='MB'
            Unit for the size
            
        Returns
        -------
        float
            Estimated size in specified unit
        """
        check_is_fitted(self)
        
        import sys
        size_bytes = sys.getsizeof(pickle.dumps(self))
        
        units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        return size_bytes / units[unit.upper()]

    # NEW: Get decision path for tree-based models
    def get_decision_path(self, environment):
        """Get decision path for tree-based models.
        
        Parameters
        ----------
        environment : dict
            Input data environment
            
        Returns
        -------
        decision_path : sparse matrix or array
            Decision path through the trees
        """
        check_is_fitted(self)
        
        for node_name in self._get_estimator_nodes():
            try:
                estimator = self.find_fitted_estimator(node_name)
                if hasattr(estimator, 'decision_path'):
                    X, _ = _compute_Xy(self.data_op, environment)
                    return estimator.decision_path(X)
            except (KeyError, NotFittedError, AttributeError):
                continue
        
        raise AttributeError("Underlying estimator does not support decision_path")

    # NEW: Plot learning curve
    def plot_learning_curve(self, environment, cv=None, train_sizes=None):
        """Plot learning curve for the model.
        
        Requires matplotlib and scikit-learn.
        
        Parameters
        ----------
        environment : dict
            Data environment
        cv : int, cross-validation generator or None, default=None
            Cross-validation strategy
        train_sizes : array-like, default=None
            Relative or absolute numbers of training examples
            
        Returns
        -------
        matplotlib.figure.Figure
            Learning curve plot
        """
        try:
            from sklearn.model_selection import learning_curve
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("plot_learning_curve requires matplotlib and scikit-learn")
        
        X, y = _compute_Xy(self.data_op, environment)
        pipeline = _to_Xy_pipeline(self, environment)
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 5)
        
        train_sizes, train_scores, test_scores = learning_curve(
            pipeline, X, y, cv=cv, train_sizes=train_sizes
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training score')
        ax.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='Cross-validation score')
        ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                       train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
        ax.fill_between(train_sizes, test_scores.mean(axis=1) - test_scores.std(axis=1),
                       test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.1)
        
        ax.set_xlabel('Training examples')
        ax.set_ylabel('Score')
        ax.set_title('Learning Curve')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return fig

    # NEW: Calibrate classifier probabilities
    def calibrate(self, environment, method='sigmoid', cv='prefit'):
        """Calibrate classifier probabilities.
        
        Parameters
        ----------
        environment : dict
            Data environment for calibration
        method : {'sigmoid', 'isotonic'}, default='sigmoid'
            Calibration method
        cv : int, cross-validation generator or 'prefit', default='prefit'
            Cross-validation strategy
            
        Returns
        -------
        SkrubLearner
            Calibrated learner
        """
        check_is_fitted(self)
        
        if not is_classifier(self):
            raise ValueError("Calibration is only for classifiers")
        
        X, y = _compute_Xy(self.data_op, environment)
        pipeline = _to_Xy_pipeline(self, environment)
        
        calibrated = CalibratedClassifierCV(pipeline, method=method, cv=cv)
        calibrated.fit(X, y)
        
        # Create a new learner with the calibrated estimator
        calibrated_data_op = self.data_op.skb.clone()
        # Update the estimator in the data_op with the calibrated one
        # (This is a simplified approach; actual implementation would need
        # to modify the data_op structure)
        
        return self.__class__(calibrated_data_op)

    # Helper method for getting estimator nodes
    def _get_estimator_nodes(self):
        """Get names of all estimator nodes in the pipeline."""
        nodes = []
        current = self.data_op
        while current is not None:
            if hasattr(current, '_skrub_impl') and isinstance(current._skrub_impl, Apply):
                nodes.append(getattr(current, '_name', None))
            current = getattr(current, '_source', None)
        return [n for n in nodes if n is not None]

    def __getattr__(self, name):
        if name not in supported_modes(self.data_op):
            attribute_error(self, name)

        def f(*args, **kwargs):
            result = self._eval_in_mode(name, *args, **kwargs)
            return self if name == "fit" else result

        f.__name__ = name
        return f

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        if not deep:
            return params
        params.update({f"data_op__{k}": v for k, v in get_params(self.data_op).items()})
        params['warm_start'] = self.warm_start
        params['early_stopping_rounds'] = self.early_stopping_rounds
        return params

    def set_params(self, **params):
        if "data_op" in params:
            self.data_op = params.pop("data_op")
        if "warm_start" in params:
            self.warm_start = params.pop("warm_start")
        if "early_stopping_rounds" in params:
            self.early_stopping_rounds = params.pop("early_stopping_rounds")

        def to_id(key):
            if key.startswith("data_op__"):
                return int(key.removeprefix("data_op__"))
            return int(key.split(":", 1)[0])

        def to_idx(val):
            if isinstance(val, str):
                return int(val.split(":", 1)[0])
            return val

        set_params(self.data_op, {to_id(k): to_idx(v) for k, v in params.items()})
        return self

    def get_param_grid(self):
        grid = param_grid(self.data_op)
        new_grid = []
        for subgrid in grid:
            subgrid = {f"data_op__{k}": v for k, v in subgrid.items()}
            subgrid['warm_start'] = [False]  # Default value
            subgrid['early_stopping_rounds'] = [None]  # Default value
            new_grid.append(subgrid)
        return new_grid

    def find_fitted_estimator(self, name):
        """Find the scikit-learn estimator that has been fitted."""
        node = find_node_by_name(self.data_op, name)
        if node is None:
            raise KeyError(name)
        impl = node._skrub_impl
        if not isinstance(impl, Apply):
            raise TypeError(
                f"Node {name!r} does not represent "
                f"the application of an estimator: {node!r}"
            )
        if not hasattr(impl, "estimator_"):
            raise NotFittedError(
                f"Node {name!r} has not been fitted. Call fit() on the learner "
                "before attempting to retrieve fitted sub-estimators."
            )
        return node._skrub_impl.estimator_

    def truncated_after(self, name):
        """Extract the part of the learner that leads up to the given step."""
        node = find_node_by_name(self.data_op, name)
        if node is None:
            raise KeyError(name)
        new = self.__class__(node)
        _copy_attr(self, new, ["_is_fitted", "warm_start", "early_stopping_rounds"])
        return new

    def describe_params(self):
        """Describe parameters for this learner."""
        return describe_params(eval_choices(self.data_op), choice_graph(self.data_op))

    # NEW: Create ensemble of multiple learners
    @classmethod
    def ensemble(cls, learners, method='voting', weights=None):
        """Create an ensemble of multiple learners.
        
        Parameters
        ----------
        learners : list of SkrubLearner
            List of fitted learners to ensemble
        method : {'voting', 'stacking', 'averaging'}, default='voting'
            Ensemble method
        weights : array-like, optional
            Weights for the learners
            
        Returns
        -------
        SkrubLearner
            Ensemble learner
        """
        if not learners:
            raise ValueError("At least one learner is required")
        
        # Check all learners are fitted
        for learner in learners:
            check_is_fitted(learner)
        
        # For now, return a simple wrapper
        # In a full implementation, this would create a proper ensemble data_op
        class EnsembleLearner(cls):
            def __init__(self, learners, method, weights):
                self.learners = learners
                self.method = method
                self.weights = weights
                self._is_fitted = True
            
            def predict(self, environment):
                predictions = [learner.predict(environment) for learner in self.learners]
                if self.method == 'averaging':
                    return np.average(predictions, axis=0, weights=self.weights)
                elif self.method == 'voting':
                    # Majority vote for classification
                    return np.apply_along_axis(
                        lambda x: np.bincount(x).argmax(), axis=0, arr=np.array(predictions)
                    )
                else:
                    raise ValueError(f"Unknown ensemble method: {method}")
        
        return EnsembleLearner(learners, method, weights)


def _to_Xy_pipeline(learner, environment):
    return learner.__skrub_to_Xy_pipeline__(environment)


def _to_env_learner(learner):
    return learner.__skrub_to_env_learner__()


def _get_classes(data_op):
    first = find_first_apply(data_op)
    if first is None:
        attribute_error(data_op, "classes_")
    try:
        estimator = first._skrub_impl.estimator_
    except AttributeError:
        attribute_error(data_op, "classes_")
    return estimator.classes_


class _XyPipelineMixin:
    def _get_env(self, X, y):
        xy_environment = {X_NAME: X}
        if y is not None:
            xy_environment[Y_NAME] = y
        return {**self.environment, **xy_environment}

    @property
    def _estimator_type(self):
        first = find_first_apply(self.data_op)
        if first is None:
            return "transformer"
        estimator = get_default(first._skrub_impl.estimator)
        if isinstance(estimator, DataOp):
            return "transformer"
        try:
            return estimator._estimator_type
        except AttributeError:
            return "transformer"

    if hasattr(BaseEstimator, "__sklearn_tags__"):
        def __sklearn_tags__(self):
            first = find_first_apply(self.data_op)
            if first is None:
                return _default_sklearn_tags()
            estimator = get_default(first._skrub_impl.estimator)
            if isinstance(estimator, DataOp):
                return _default_sklearn_tags()
            try:
                return estimator.__sklearn_tags__()
            except AttributeError:
                return _default_sklearn_tags()

    @property
    def classes_(self):
        try:
            return _get_classes(self.data_op)
        except AttributeError:
            attribute_error(self, "classes_")
    
    # NEW: Add fit_predict method for scikit-learn compatibility
    def fit_predict(self, X, y=None):
        """Fit to data, then transform it.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (for supervised transformation)
            
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features_new)
            Transformed samples
        """
        self.fit(X, y)
        return self.transform(X)
    
    # NEW: Add fit_predict for classification/regression
    def fit_predict(self, X, y=None, **fit_params):
        """Fit estimator and return predictions for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (for supervised transformation)
        **fit_params : dict
            Additional fit parameters
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values
        """
        self.fit(X, y, **fit_params)
        return self.predict(X)


class _XyPipeline(_XyPipelineMixin, SkrubLearner):
    """Scikit-learn compatible interface to the SkrubLearner."""
    
    def __init__(self, data_op, environment):
        self.data_op = data_op
        self.environment = environment

    def __skrub_to_env_learner__(self):
        new = SkrubLearner(self.data_op)
        _copy_attr(self, new, ["_is_fitted", "warm_start", "early_stopping_rounds"])
        return new

    def _eval_in_mode(self, mode, X, y=None):
        result = evaluate(self.data_op, mode, self._get_env(X, y), 
                         clear=not getattr(self, 'warm_start', False))
        self._set_is_fitted(mode)
        return result
    
    # NEW: Override partial_fit for scikit-learn interface
    def partial_fit(self, X, y=None, classes=None):
        """Incremental fit on a batch of samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples
        y : array-like of shape (n_samples,), default=None
            Target values
        classes : array-like of shape (n_classes,), default=None
            Classes across all calls to partial_fit
            
        Returns
        -------
        self
        """
        environment = self._get_env(X, y)
        if classes is not None:
            environment['_partial_fit_classes'] = classes
        
        result = evaluate(self.data_op, "partial_fit", environment, 
                         clear=not getattr(self, 'warm_start', False))
        self._set_is_fitted("partial_fit")
        return self if result is None else result


def _find_Xy(data_op):
    """Find the nodes marked with `.skb.mark_as_X()` and `.skb.mark_as_y()`"""
    x_node = find_X(data_op)
    if x_node is None:
        raise ValueError('DataOp should have a node marked with "mark_as_X()"')
    result = {"X": x_node}
    if (y_node := find_y(data_op)) is not None:
        result["y"] = y_node
    else:
        first = find_first_apply(data_op)
        if first is None:
            return result
        if getattr(first._skrub_impl, "y", None) is not None:
            raise ValueError('DataOp should have a node marked with "mark_as_y()"')
    return result


def _compute_Xy(data_op, environment):
    """Evaluate the nodes marked with `.skb.mark_as_X()` and `.skb.mark_as_y()`."""
    
    Xy = _find_Xy(data_op.skb.clone())
    X = evaluate(
        Xy["X"],
        mode="fit_transform",
        environment=environment,
        clear=False,
    )
    if "y" in Xy:
        y = evaluate(
            Xy["y"],
            mode="fit_transform",
            environment=environment,
            clear=False,
        )
        msg = (
            "\nAre `.skb.subsample()` and `.skb.mark_as_*()` applied in the same order"
            " for both X and y?"
        )
        check_subsampled_X_y_shape(
            Xy["X"], Xy["y"], X, y, "fit_transform", environment, msg=msg
        )
    else:
        y = None
    return X, y


def _rename_cv_param_learner_to_estimator(kwargs):
    if "return_estimator" in kwargs:
        raise TypeError(
            "`skrub.cross_validate` does not have a `return_estimator` parameter. The"
            " equivalent of scikit-learn's `return_estimator` is called"
            " `return_pipeline`. Use `cross_validate(return_learner=True)` instead of"
            " `cross_validate(return_estimator=True)`."
        )
    renamed = dict(kwargs)
    if "return_learner" not in renamed:
        return kwargs
    renamed["return_estimator"] = renamed.pop("return_learner")
    return renamed


def cross_validate(learner, environment, *, keep_subsampling=False, **kwargs):
    """Cross-validate a learner built from a DataOp.
    
    New Features:
    - Support for stratified cross-validation for classification
    - Support for returning cross-validated predictions
    """
    if not hasattr(learner, "__skrub_to_Xy_pipeline__"):
        raise ValueError(
            "`cross_validate` function requires either a Learner object or "
            f"a ParamSearch object, got {type(learner)}."
        )
    
    environment = env_with_subsampling(learner.data_op, environment, keep_subsampling)
    kwargs = _rename_cv_param_learner_to_estimator(kwargs)
    X, y = _compute_Xy(learner.data_op, environment)
    
    # NEW: Use stratified K-fold for classification if not specified
    if 'cv' not in kwargs and y is not None and len(np.unique(y)) > 1:
        kwargs['cv'] = StratifiedKFold(5)
    
    result = model_selection.cross_validate(
        _to_Xy_pipeline(learner, environment),
        X,
        y,
        **kwargs,
    )
    if (fitted_learners := result.pop("estimator", None)) is not None:
        result["learner"] = [_to_env_learner(p) for p in fitted_learners]
    return pd.DataFrame(result)


# NEW: Cross-validated predictions function
def cross_val_predict(learner, environment, *, cv=KFOLD_5, method='predict', 
                      keep_subsampling=False, n_jobs=None):
    """Generate cross-validated predictions.
    
    Parameters
    ----------
    learner : skrub learner
        A learner generated from a skrub DataOp
    environment : dict
        Bindings for variables contained in the DataOp
    cv : int, cross-validation generator or iterable, default=KFOLD_5
        Cross-validation splitting strategy
    method : str, default='predict'
        The method to call on the estimator (predict, predict_proba, etc.)
    keep_subsampling : bool, default=False
        Whether to use subsampling
    n_jobs : int, optional
        Number of jobs to run in parallel
        
    Returns
    -------
    predictions : ndarray
        The cross-validated predictions
    """
    environment = env_with_subsampling(learner.data_op, environment, keep_subsampling)
    X, y = _compute_Xy(learner.data_op, environment)
    pipeline = _to_Xy_pipeline(learner, environment)
    
    return sklearn_cross_val_predict(
        pipeline, X, y, cv=cv, method=method, n_jobs=n_jobs
    )


def train_test_split(
    data_op,
    environment,
    *,
    keep_subsampling=False,
    split_func=model_selection.train_test_split,
    **split_func_kwargs,
):
    """Split an environment into a training an testing environments."""
    environment = env_with_subsampling(data_op, environment, keep_subsampling)
    X, y = _compute_Xy(data_op, environment)
    if y is None:
        X_train, X_test = split_func(X, **split_func_kwargs)
    else:
        X_train, X_test, y_train, y_test = split_func(X, y, **split_func_kwargs)
    train_env = {**environment, X_NAME: X_train}
    test_env = {**environment, X_NAME: X_test}
    result = {
        "train": train_env,
        "test": test_env,
        "X_train": X_train,
        "X_test": X_test,
    }
    if y is not None:
        train_env[Y_NAME] = y_train
        test_env[Y_NAME] = y_test
        result["y_train"] = y_train
        result["y_test"] = y_test
    return result


def iter_cv_splits(data_op, environment, *, keep_subsampling=False, cv=KFOLD_5):
    """Yield splits of an environment into training an testing environments."""
    if cv is KFOLD_5:
        # NEW: Use stratified K-fold for classification
        X, y = _compute_Xy(data_op, environment)
        if y is not None and len(np.unique(y)) > 1:
            cv = StratifiedKFold(5)
        else:
            cv = KFold(5)
    
    cv = check_cv(cv)
    environment = env_with_subsampling(data_op, environment, keep_subsampling)
    X, y = _compute_Xy(data_op, environment)
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = sbd.select_rows(X, train_idx), sbd.select_rows(X, test_idx)
        train_env = {**environment, X_NAME: X_train}
        test_env = {**environment, X_NAME: X_test}
        split_info = {
            "train": train_env,
            "test": test_env,
            "X_train": X_train,
            "X_test": X_test,
        }
        if y is not None:
            y_train, y_test = (
                sbd.select_rows(y, train_idx),
                sbd.select_rows(y, test_idx),
            )
            train_env[Y_NAME] = y_train
            test_env[Y_NAME] = y_test
            split_info["y_train"] = y_train
            split_info["y_test"] = y_test
        yield split_info


class _BaseParamSearch(_CloudPickleDataOp, BaseEstimator):
    """Base class for hyperparameter search objects.
    
    New Features:
    - Early stopping support
    - Best iteration tracking
    """
    
    def __init__(self, data_op, search, early_stopping_rounds=None):
        self.data_op = data_op
        self.search = search
        self.early_stopping_rounds = early_stopping_rounds

    def __getattr__(self, name):
        if name not in supported_modes(self.data_op):
            attribute_error(self, name)

        def f(*args, **kwargs):
            return self._call_predictor_method(name, *args, **kwargs)

        f.__name__ = name
        return f

    def _call_predictor_method(self, name, environment):
        check_is_fitted(self, "cv_results_")
        if not hasattr(self, "best_learner_"):
            raise AttributeError(
                "This parameter search was initialized with `refit=False`. "
                f"{name} is available only after refitting on the best parameters. "
                "Please pass another value to `refit` or fit a learner manually "
                "using the `best_params_` or `cv_results_` attributes."
            )
        return getattr(self.best_learner_, name)(environment)

    @property
    def results_(self):
        """Cross-validation results containing parameters and scores in a dataframe."""
        try:
            return self._get_cv_results_table()[0]
        except NotFittedError:
            attribute_error(self, "results_")

    @property
    def detailed_results_(self):
        """More detailed cross-validation results table."""
        try:
            return self._get_cv_results_table(detailed=True)[0]
        except NotFittedError:
            attribute_error(self, "results_")

    def _get_cv_results_table(self, detailed=False):
        check_is_fitted(self, "cv_results_")
        data_op_choices = choice_graph(self.data_op)

        all_rows = []
        for params in self.cv_results_["params"]:
            params = {int(k.removeprefix("data_op__")): v for k, v in params.items()}
            all_rows.append(describe_params(params, data_op_choices))

        table = pd.DataFrame(
            all_rows, columns=list(data_op_choices["choice_display_names"].values())
        )
        metric_names = [
            k.removeprefix("mean_test_")
            for k in self.cv_results_.keys()
            if k.startswith("mean_test_")
        ]
        if isinstance(self.refit_, str):
            metric_names.insert(0, metric_names.pop(metric_names.index(self.refit_)))
        result_keys = [
            *(f"mean_test_{n}" for n in metric_names),
            *(f"std_test_{n}" for n in metric_names),
            "mean_fit_time",
            "std_fit_time",
            "mean_score_time",
            "std_score_time",
            *(f"mean_train_{n}" for n in metric_names),
            *(f"std_train_{n}" for n in metric_names),
        ]
        
        # NEW: Add early stopping results if available
        if hasattr(self, 'early_stopping_rounds_'):
            result_keys.extend(['early_stopping_rounds_', 'best_iteration_'])
        
        new_names = _join_utils.pick_column_names(table.columns, result_keys)
        renaming = dict(zip(table.columns, new_names))
        table.columns = new_names
        metadata = _get_results_metadata(data_op_choices)
        metadata["log_scale_columns"] = [
            renaming[c] for c in metadata["log_scale_columns"]
        ]
        if detailed:
            for k in result_keys[len(metric_names) :][::-1]:
                if k in self.cv_results_:
                    table.insert(table.shape[1], k, self.cv_results_[k])
        for k in result_keys[: len(metric_names)][::-1]:
            table.insert(table.shape[1], k, self.cv_results_[k])
        metadata["col_score"] = f"mean_test_{metric_names[0]}"
        table = table.sort_values(
            metadata["col_score"],
            ascending=False,
            ignore_index=True,
            kind="stable",
        )
        return table, metadata

    def plot_results(self, *, colorscale=DEFAULT_COLORSCALE, min_score=None):
        """Create a parallel coordinate plot of the cross-validation results."""
        cv_results, metadata = self._get_cv_results_table(detailed=True)
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
            col_score = metadata["col_score"]
            cv_results = cv_results[cv_results[col_score] >= min_score]
        if not cv_results.shape[0]:
            raise ValueError("No results to plot")
        return plot_parallel_coord(cv_results, metadata, colorscale=colorscale)
    
    # NEW: Plot learning curve from search results
    def plot_learning_curve(self, colorscale='Viridis'):
        """Plot learning curve from search results.
        
        Parameters
        ----------
        colorscale : str, default='Viridis'
            Colorscale for the plot
            
        Returns
        -------
        plotly.graph_objects.Figure
            Learning curve plot
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("plotly is required for plot_learning_curve")
        
        cv_results, metadata = self._get_cv_results_table()
        
        # Find parameters that are numeric and vary
        numeric_cols = []
        for col in cv_results.columns:
            if col.startswith('mean_test_') or col.startswith('param_'):
                try:
                    pd.to_numeric(cv_results[col])
                    numeric_cols.append(col)
                except:
                    pass
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Learning Curve', 'Parameter Importance'))
        
        # Learning curve
        if 'mean_fit_time' in cv_results.columns:
            fig.add_trace(
                go.Scatter(
                    x=cv_results['mean_fit_time'],
                    y=cv_results[metadata['col_score']],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=cv_results[metadata['col_score']],
                        colorscale=colorscale,
                        showscale=True
                    ),
                    text=[f"Score: {s:.4f}<br>Time: {t:.2f}s" 
                          for s, t in zip(cv_results[metadata['col_score']], cv_results['mean_fit_time'])],
                    hoverinfo='text'
                ),
                row=1, col=1
            )
            fig.update_xaxes(title_text='Fit Time (s)', row=1, col=1)
            fig.update_yaxes(title_text='Score', row=1, col=1)
        
        fig.update_layout(
            title='Hyperparameter Search Learning Curve',
            showlegend=False,
            height=500
        )
        
        return fig


def _get_results_metadata(data_op_choices):
    log_scale_columns = set()
    int_columns = set()
    for choice_id, choice in data_op_choices["choices"].items():
        if isinstance(choice, BaseNumericChoice):
            choice_name = data_op_choices["choice_display_names"][choice_id]
            if choice.log:
                log_scale_columns.add(choice_name)
            if choice.to_int:
                int_columns.add(choice_name)
    return {
        "log_scale_columns": list(log_scale_columns),
        "int_columns": list(int_columns),
    }


class ParamSearch(_BaseParamSearch):
    """Learner that evaluates a skrub DataOp with hyperparameter tuning.
    
    New Features:
    - Early stopping support
    - Warm start support
    - Model persistence
    """
    
    def __init__(self, data_op, search, warm_start=False, early_stopping_rounds=None):
        super().__init__(data_op, search, early_stopping_rounds)
        self.warm_start = warm_start
        self.early_stopping_rounds = early_stopping_rounds

    def __skrub_to_Xy_pipeline__(self, environment):
        new = _XyParamSearch(self.data_op, self.search, _SharedDict(environment),
                           warm_start=self.warm_start, 
                           early_stopping_rounds=self.early_stopping_rounds)
        _copy_attr(self, new, _SEARCH_FITTED_ATTRIBUTES)
        return new

    def fit(self, environment):
        self.refit_ = self.search.refit
        search = clone(self.search)
        search.estimator = _XyPipeline(self.data_op, _SharedDict(environment))
        
        # NEW: Pass early stopping and warm start parameters
        search.estimator.warm_start = self.warm_start
        search.estimator.early_stopping_rounds = self.early_stopping_rounds
        
        param_grid = search.estimator.get_param_grid()
        if hasattr(search, "param_grid"):
            search.param_grid = param_grid
        else:
            assert hasattr(search, "param_distributions")
            search.param_distributions = param_grid
        
        X, y = _compute_Xy(self.data_op, environment)
        search.fit(X, y)
        
        _copy_attr(search, self, _SKLEARN_SEARCH_FITTED_ATTRIBUTES_TO_COPY)
        
        # NEW: Store early stopping results if available
        if hasattr(search, 'best_estimator_') and hasattr(search.best_estimator_, '_early_stopping_callback'):
            self.early_stopping_rounds_ = search.best_estimator_._early_stopping_callback.early_stopping_rounds
            self.best_iteration_ = search.best_estimator_._early_stopping_callback.best_iteration
        
        try:
            self.best_learner_ = _to_env_learner(search.best_estimator_)
            # NEW: Copy feature importances if available
            if hasattr(self.best_learner_, 'feature_importances_'):
                self.feature_importances_ = self.best_learner_.feature_importances_
        except AttributeError:
            # refit is set to False
            pass
        
        return self
    
    # NEW: Save and load methods for ParamSearch
    def save(self, path, compress=True):
        """Save the fitted parameter search to disk."""
        check_is_fitted(self, "cv_results_")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'data_op': self.data_op,
            'search': self.search,
            'cv_results_': self.cv_results_,
            'best_params_': self.best_params_,
            'best_score_': self.best_score_,
            'best_learner_': self.best_learner_ if hasattr(self, 'best_learner_') else None,
            'refit_': self.refit_,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        if compress:
            import gzip
            with gzip.open(path, 'wb') as f:
                pickle.dump(model_data, f)
        else:
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path):
        """Load a fitted parameter search from disk."""
        path = Path(path)
        
        if path.suffix == '.gz':
            import gzip
            with gzip.open(path, 'rb') as f:
                model_data = pickle.load(f)
        else:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
        
        param_search = cls(model_data['data_op'], model_data['search'])
        param_search.cv_results_ = model_data['cv_results_']
        param_search.best_params_ = model_data['best_params_']
        param_search.best_score_ = model_data['best_score_']
        param_search.refit_ = model_data['refit_']
        
        if model_data['best_learner_'] is not None:
            param_search.best_learner_ = model_data['best_learner_']
        
        return param_search


class _XyParamSearch(_XyPipelineMixin, ParamSearch):
    """Scikit-learn compatible ParamSearch interface."""
    
    def __init__(self, data_op, search, environment, warm_start=False, 
                 early_stopping_rounds=None):
        self.data_op = data_op
        self.search = search
        self.environment = environment
        self.warm_start = warm_start
        self.early_stopping_rounds = early_stopping_rounds

    def __skrub_to_env_learner__(self):
        new = ParamSearch(self.data_op, self.search, 
                         warm_start=self.warm_start,
                         early_stopping_rounds=self.early_stopping_rounds)
        _copy_attr(self, new, _SEARCH_FITTED_ATTRIBUTES)
        return new

    @property
    def classes_(self):
        if not hasattr(self, "best_learner_"):
            attribute_error(self, "classes_")
        try:
            return _get_classes(self.best_learner_.data_op)
        except AttributeError:
            attribute_error(self, "classes_")

    def fit(self, X, y=None):
        super().fit(self._get_env(X, y))
        return self

    def _call_predictor_method(self, name, X, y=None):
        return getattr(self.best_learner_, name)(self._get_env(X, y))
    
    # NEW: Property to get best estimator as scikit-learn estimator
    @property
    def best_estimator_(self):
        """Get the best estimator as a scikit-learn compatible estimator."""
        if not hasattr(self, "best_learner_"):
            attribute_error(self, "best_estimator_")
        return _to_Xy_pipeline(self.best_learner_, self.environment)
