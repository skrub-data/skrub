# Scikit-learn-ish interface to the skrub DataOps

import pandas as pd
from sklearn import model_selection
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, check_cv
from sklearn.utils.validation import check_is_fitted

from .. import _dataframe as sbd
from .. import _join_utils
from ._choosing import BaseNumericChoice, get_default
from ._data_ops import Apply, DataOp, check_subsampled_X_y_shape
from ._evaluation import (
    choice_graph,
    chosen_or_default_outcomes,
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

_FITTING_METHODS = ["fit", "fit_transform"]
_SKLEARN_SEARCH_FITTED_ATTRIBUTES_TO_COPY = [
    # Some attributes are intentionally left out because the skrub ParamSearch
    # doesn't expose them or they need more than simply copying to the skrub
    # object and are handled separately.
    #
    # In particular this list does not include `best_estimator_` nor `classes_`.
    #
    "cv_results_",
    "best_score_",
    "best_params_",
    "best_index_",
    "scorer_",
    "n_splits_",
    "refit_time_",
    "multimetric_",
]
_SEARCH_FITTED_ATTRIBUTES = _SKLEARN_SEARCH_FITTED_ATTRIBUTES_TO_COPY + [
    "best_learner_"
]


def _default_sklearn_tags():
    class _DummyTransformer(TransformerMixin, BaseEstimator):
        pass

    return _DummyTransformer().__sklearn_tags__()


class _SharedDict(dict):
    """A dict that does not get copied during deepcopy/sklearn clone.

    To make the evaluation environment available to the learners' methods, we
    put it in an attribute of the _XyPipeline or _XyParamSearch objects (see
    _XyPipeline for details). As it is potentially large and is never modified
    we avoid making copies by overriding __deepcopy__ and __sklearn_clone__.
    """

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
    """
    Mixin to serialize the `DataOp` attribute with cloudpickle when pickling a
    learner.
    """

    _cloudpickle_attributes = ["data_op"]


class SkrubLearner(_CloudPickleDataOp, BaseEstimator):
    """Learner that evaluates a skrub DataOp.

    This class is not meant to be instantiated manually, ``SkrubLearner``
    objects are created by calling :meth:`DataOp.skb.make_learner()` on a
    DataOp.
    """

    def __init__(self, data_op):
        self.data_op = data_op

    def __skrub_to_Xy_pipeline__(self, environment):
        """Convert to a fully scikit-learn compatible pipeline (fit takes X, y)."""
        new = _XyPipeline(self.data_op, _SharedDict(environment))
        _copy_attr(self, new, ["_is_fitted"])
        return new

    def _set_is_fitted(self, mode):
        if mode in _FITTING_METHODS:
            self._is_fitted = True

    def __sklearn_is_fitted__(self):
        return getattr(self, "_is_fitted", False)

    def _eval_in_mode(self, mode, environment):
        if mode not in _FITTING_METHODS:
            check_is_fitted(self)
        result = evaluate(self.data_op, mode, environment, clear=True)
        self._set_is_fitted(mode)
        return result

    def report(self, *, environment, mode, **full_report_kwargs):
        """Call the method specified by ``mode`` and return the result and full report.

        See :meth:`DataOp.skb.full_report` for more information.

        Parameters
        ----------
        environment : dict
            Bindings for variables contained in the :class:`DataOp` that was
            used to create this learner
            (e.g. ``{"X": X_df, "other_table": df, ...}``).
        mode : str
            The method to call in order to generate the report, such as
            ``"fit"``, ``"predict"``, etc.
        full_report_kwargs : dict
            See :meth:`DataOp.skb.full_report`

        Returns
        -------
        dict
            The result of ``DataOp.skb.full_report``: a dict containing
            ``'result'``, ``'error'`` and ``'report_path'``.
        """
        from ._inspection import full_report

        if mode not in _FITTING_METHODS:
            check_is_fitted(self)

        full_report_kwargs["clear"] = True
        result = full_report(
            self.data_op, environment=environment, mode=mode, **full_report_kwargs
        )
        if mode == "fit" and result["result"] is not None:
            result["result"] = self
        self._set_is_fitted(mode)
        return result

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
        return params

    def set_params(self, **params):
        if "data_op" in params:
            self.data_op = params.pop("data_op")
        set_params(
            self.data_op, {int(k.lstrip("data_op__")): v for k, v in params.items()}
        )
        return self

    def get_param_grid(self):
        grid = param_grid(self.data_op)
        new_grid = []
        for subgrid in grid:
            subgrid = {f"data_op__{k}": v for k, v in subgrid.items()}
            new_grid.append(subgrid)
        return new_grid

    def find_fitted_estimator(self, name):
        """
        Find the scikit-learn estimator that has been fitted in a ``.skb.apply()`` step.

        This can be useful for example to inspect the fitted attributes of the
        estimator. The ``apply`` step must have been given a name with
        ``.skb.set_name()`` (see examples below).

        Parameters
        ----------
        name : str
            The name of the ``.skb.apply()`` step in which an estimator has
            been fitted.

        Returns
        -------
        scikit-learn estimator
            The fitted estimator. Depending on the nature of the estimator it
            may be wrapped in a ``skrub.ApplyToCols`` or ``skrub.ApplyToFrame``,
            see examples below.

        See also
        --------
        skrub.DataOp.skb.set_name :
            Give a name to this DataOp.

        skrub.DataOp.skb.apply :
            Apply a scikit-learn estimator to a dataframe or numpy array.

        Examples
        --------
        >>> from sklearn.decomposition import PCA
        >>> from sklearn.dummy import DummyClassifier
        >>> import skrub
        >>> from skrub import selectors as s

        >>> orders = skrub.datasets.toy_orders()
        >>> X, y = skrub.X(), skrub.y()
        >>> pred = (
        ...     X.skb.apply(skrub.StringEncoder(n_components=2), cols=["product"])
        ...     .skb.set_name("product_encoder")
        ...     .skb.apply(skrub.ToDatetime(), cols=["date"])
        ...     .skb.apply(skrub.DatetimeEncoder(add_total_seconds=False), cols=["date"])
        ...     .skb.apply(PCA(n_components=2), cols=s.glob("date_*"))
        ...     .skb.set_name("pca")
        ...     .skb.apply(DummyClassifier(), y=y)
        ...     .skb.set_name("classifier")
        ... )
        >>> learner = pred.skb.make_learner()
        >>> learner.fit({'X': orders.X, 'y': orders.y})
        SkrubLearner(data_op=<classifier | Apply DummyClassifier>)

        We can retrieve the fitted transformer for a given step with
        ``find_fitted_estimator``:

        >>> learner.find_fitted_estimator("classifier")
        DummyClassifier()

        Depending on the parameters passed to :meth:`DataOp.skb.apply`, the
        estimator we provide can be wrapped in a skrub transformer that applies
        it to several columns in the input, or to a subset of the columns in a
        dataframe. In other cases it may be applied without any wrapping. We
        provide examples for those 3 different cases below.

        Case 1: the ``StringEncoder`` is a skrub single-column transformer: it
        transforms a single column. In the learner it gets wrapped in a
        :class:`ApplyToCols` which independently fits a separate instance of the
        ``StringEncoder`` to each of the columns it transforms (in this case there is
        only one column, ``'product'``). The individual transformers can be found in the
        fitted attribute ``transformers_`` which maps column names to the corresponding
        fitted transformer.

        >>> encoder = learner.find_fitted_estimator('product_encoder')
        >>> encoder.transformers_
        {'product': StringEncoder(n_components=2)}
        >>> encoder.transformers_['product'].vectorizer_.vocabulary_
        {' pe': 2, 'pen': 12, 'en ': 8, ' pen': 3, 'pen ': 13, ' cu': 0, 'cup': 6, 'up ': 18, ' cup': 1, 'cup ': 7, ' sp': 4, 'spo': 16, 'poo': 14, 'oon': 10, 'on ': 9, ' spo': 5, 'spoo': 17, 'poon': 15, 'oon ': 11}

        This case (wrapping in :class:`ApplyToCols`) happens when the estimator is a skrub
        single-column transformer (it has a ``__single_column_transformer__``
        attribute), we pass ``.skb.apply(how='cols')`` or we pass
        ``.skb.apply(allow_reject=True)``.

        Case 2: the ``PCA`` is a regular scikit-learn transformer. In the learner it
        gets wrapped in a :class:`ApplyToFrame` which applies it to the subset of columns
        in the dataframe selected by the ``cols`` argument passed to ``.skb.apply()``.
        The fitted ``PCA`` can be found in the fitted attribute ``transformer_``.

        >>> pca = learner.find_fitted_estimator('pca')
        >>> pca
        ApplyToFrame(cols=glob('date_*'), transformer=PCA(n_components=2))
        >>> pca.transformer_
        PCA(n_components=2)
        >>> pca.transformer_.mean_
        array([2020.,    4.,    4.], dtype=float32)

        This case (wrapping in :class:`ApplyToFrame`) happens when the estimator is a
        scikit-learn transformer but not a single-column transformer, or we
        pass ``.skb.apply(how='frame')``.

        The ``DummyRegressor`` is a scikit-learn predictor. In the learner it gets
        applied directly to the input dataframe without any wrapping.

        >>> classifier = learner.find_fitted_estimator('classifier')
        >>> classifier
        DummyClassifier()
        >>> classifier.class_prior_
        array([0.75, 0.25])

        This case (no wrapping) happens when the estimator is a scikit-learn predictor
        (not a transformer), the input is not a dataframe (e.g. it is a numpy array), or
        we pass ``.skb.apply(how='no_wrap')``.
        """  # noqa: E501
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
        """Extract the part of the learner that leads up to the given step.

        This is similar to slicing a scikit-learn pipeline. It can be useful
        for example to drive the hyperparameter selection with a supervised
        task but then extract only the part of the learner that performs
        feature extraction.

        The target step must have been given a name with ``.skb.set_name()``.

        Parameters
        ----------
        name : str
            The name of the intermediate step we want to extract.

        Returns
        -------
        SkrubLearner
            A skrub learner that performs all the transformations leading up to
            (and including) the required step.

        Examples
        --------
        >>> from sklearn.dummy import DummyClassifier
        >>> import skrub

        >>> orders = skrub.datasets.toy_orders()
        >>> X, y = skrub.X(), skrub.y()
        >>> pred = (
        ...     X.skb.apply(
        ...         skrub.TableVectorizer(datetime=skrub.DatetimeEncoder(add_total_seconds=False))
        ...     )
        ...     .skb.set_name("vectorizer")
        ...     .skb.apply(DummyClassifier(), y=y)
        ... )
        >>> learner = pred.skb.make_learner()
        >>> learner.fit({"X": orders.X, "y": orders.y})
        SkrubLearner(data_op=<Apply DummyClassifier>)
        >>> learner.predict({"X": orders.X})
        array([False, False, False, False])

        Truncate the learner after vectorization:

        >>> vectorizer = learner.truncated_after("vectorizer")
        >>> vectorizer
        SkrubLearner(data_op=<vectorizer | Apply TableVectorizer>)
        >>> vectorizer.transform({"X": orders.X})
            ID  product_cup  product_pen  ...  date_year  date_month  date_day
        0  1.0          0.0          1.0  ...     2020.0         4.0       3.0
        1  2.0          1.0          0.0  ...     2020.0         4.0       4.0
        2  3.0          1.0          0.0  ...     2020.0         4.0       4.0
        3  4.0          0.0          0.0  ...     2020.0         4.0       5.0

        Note this differs from ``find_fitted_estimator`` which extracts the inner
        scikit-learn estimator that has been fitted inside of a single step.

        This contains the full transformation up to the given step:

        >>> learner.truncated_after("vectorizer")
        SkrubLearner(data_op=<vectorizer | Apply TableVectorizer>)

        The result of ``find_fitted_estimator`` only contains the inner
        ``TableVectorizer`` that was fitted inside of the ``"vectorizer"``
        step:

        >>> learner.find_fitted_estimator("vectorizer")
        ApplyToFrame(transformer=TableVectorizer(datetime=DatetimeEncoder(add_total_seconds=False)))
        """  # noqa: E501
        node = find_node_by_name(self.data_op, name)
        if node is None:
            raise KeyError(name)
        new = self.__class__(node)
        _copy_attr(self, new, ["_is_fitted"])
        return new

    def describe_params(self):
        """Describe parameters for this learner.

        Returns a human-readable description (in form of a dict) of the
        parameters (outcomes of `choose_*` objects contained in the
        DataOp).
        """
        return describe_params(
            chosen_or_default_outcomes(self.data_op), choice_graph(self.data_op)
        )


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
        # scikit-learn >= 1.6

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


class _XyPipeline(_XyPipelineMixin, SkrubLearner):
    """
    Scikit-learn compatible interface to the SkrubLearner.

    This is a private, transient class used only during cross-validation.

    It is used to swap out the fit({'baskets': ..., 'products': ...}) interface
    with the scikit-learn fit(X, y). It exists only during cross-validation and
    is converted back to the skrub learner interface after (if the fitted
    learners are kept, e.g., in the `best_learner_`). It shares state (the
    `DataOp`) with the `SkrubLearner` that it is converted to and from.

    To make the evaluation environment (the {'baskets': ...} dict) available,
    that is stored as an attribute of type SharedDict (a dict type for which
    scikit-learn's clone incurs no copy).
    """

    def __init__(self, data_op, environment):
        self.data_op = data_op
        self.environment = environment

    def __skrub_to_env_learner__(self):
        new = SkrubLearner(self.data_op)
        _copy_attr(self, new, ["_is_fitted"])
        return new

    def _eval_in_mode(self, mode, X, y=None):
        result = evaluate(self.data_op, mode, self._get_env(X, y), clear=True)
        self._set_is_fitted(mode)
        return result


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
            # the estimator requests a y so some node must have been
            # marked as y
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

    This runs cross-validation from a learner that was built from a skrub
    DataOp with :func:`DataOp.skb.make_learner`, :func:`DataOp.skb.make_grid_search` or
    :func:`DataOp.skb.make_randomized_search`.

    It is useful to run nested cross-validation of a grid search or randomized
    search.

    Parameters
    ----------
    learner : skrub learner
        A learner generated from a skrub DataOp.

    environment : dict
        Bindings for variables contained in the DataOp.

    keep_subsampling : bool, default=False
        If True, and if subsampling has been configured (see
        :meth:`DataOp.skb.subsample`), use a subsample of the data. By
        default subsampling is not applied and all the data is used.

    kwargs : dict
        All other named arguments are forwarded to
        :func:`sklearn.model_selection.cross_validate`, except that scikit-learn's
        ``return_estimator`` parameter is named ``return_learner`` here.

    Returns
    -------
    dict
        Cross-validation results.

    See also
    --------
    :func:`sklearn.model_selection.cross_validate`:
        Evaluate metric(s) by cross-validation and also record fit/score times.

    :func:`skrub.DataOp.skb.make_learner`:
        Get a skrub learner for this DataOp.

    :func:`skrub.DataOp.skb.make_grid_search`:
        Find the best parameters with grid search.

    :func:`skrub.DataOp.skb.make_randomized_search`:
        Find the best parameters with grid search.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> import skrub

    >>> X_a, y_a = make_classification(random_state=0)
    >>> X, y = skrub.X(X_a), skrub.y(y_a)
    >>> log_reg = LogisticRegression(
    ...     **skrub.choose_float(0.01, 1.0, log=True, name="C")
    ... )
    >>> pred = X.skb.apply(log_reg, y=y)
    >>> search = pred.skb.make_randomized_search(random_state=0)
    >>> skrub.cross_validate(search, pred.skb.get_data())['test_score'] # doctest: +SKIP
    0    0.75
    1    0.90
    2    0.95
    3    0.75
    4    0.85
    Name: test_score, dtype: float64
    """
    environment = env_with_subsampling(learner.data_op, environment, keep_subsampling)
    kwargs = _rename_cv_param_learner_to_estimator(kwargs)
    X, y = _compute_Xy(learner.data_op, environment)
    result = model_selection.cross_validate(
        _to_Xy_pipeline(learner, environment),
        X,
        y,
        **kwargs,
    )
    if (fitted_learners := result.pop("estimator", None)) is not None:
        result["learner"] = [_to_env_learner(p) for p in fitted_learners]
    return pd.DataFrame(result)


def train_test_split(
    data_op,
    environment,
    *,
    keep_subsampling=False,
    split_func=model_selection.train_test_split,
    **split_func_kwargs,
):
    """Split an environment into a training an testing environments.

    This functionality is exposed to users through the
    ``DataOp.skb.train_test_split()`` method. See the corresponding docstring for
    details and examples.
    """
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
    """Yield splits of an environment into training an testing environments.

    This functionality is exposed to users through the
    ``DataOp.skb.iter_cv_splits()`` method. See the corresponding docstring for
    details and examples.
    """
    if cv is KFOLD_5:
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


class ParamSearch(_CloudPickleDataOp, BaseEstimator):
    """Learner that evaluates a skrub DataOp with hyperparameter tuning.

    This class is not meant to be instantiated manually, ``ParamSearch``
    objects are created by calling :meth:`DataOp.skb.make_grid_search()` or
    :meth:`DataOp.skb.make_randomized_search()` on a DataOp.
    """

    def __init__(self, data_op, search):
        self.data_op = data_op
        self.search = search

    def __skrub_to_Xy_pipeline__(self, environment):
        new = _XyParamSearch(self.data_op, self.search, _SharedDict(environment))
        _copy_attr(self, new, _SEARCH_FITTED_ATTRIBUTES)
        return new

    def fit(self, environment):
        search = clone(self.search)
        search.estimator = _XyPipeline(self.data_op, _SharedDict(environment))
        param_grid = search.estimator.get_param_grid()
        if hasattr(search, "param_grid"):
            search.param_grid = param_grid
        else:
            assert hasattr(search, "param_distributions")
            search.param_distributions = param_grid
        X, y = _compute_Xy(self.data_op, environment)
        search.fit(X, y)
        _copy_attr(search, self, _SKLEARN_SEARCH_FITTED_ATTRIBUTES_TO_COPY)
        try:
            self.best_learner_ = _to_env_learner(search.best_estimator_)
        except AttributeError:
            # refit is set to False, there is no best_estimator_
            pass
        return self

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
        """
        Cross-validation results containing parameters and scores in a dataframe.
        """
        try:
            return self._get_cv_results_table()[0]
        except NotFittedError:
            attribute_error(self, "results_")

    @property
    def detailed_results_(self):
        """
        More detailed cross-validation results table.

        Similar to ``results_`` but also contains the standard deviation of
        scores across folds, the scores on training data, and the fit and
        score durations.
        """
        try:
            return self._get_cv_results_table(detailed=True)[0]
        except NotFittedError:
            attribute_error(self, "results_")

    def _get_cv_results_table(self, detailed=False):
        check_is_fitted(self, "cv_results_")
        data_op_choices = choice_graph(self.data_op)

        all_rows = []
        for params in self.cv_results_["params"]:
            params = {int(k.lstrip("data_op__")): v for k, v in params.items()}
            all_rows.append(describe_params(params, data_op_choices))

        table = pd.DataFrame(
            all_rows, columns=list(data_op_choices["choice_display_names"].values())
        )
        if isinstance(self.scorer_, dict):
            metric_names = list(self.scorer_.keys())
            if isinstance(self.search.refit, str):
                metric_names.insert(
                    0, metric_names.pop(metric_names.index(self.search.refit))
                )
        else:
            metric_names = ["score"]
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
        """Create a parallel coordinate plot of the cross-validation results.

        Plotly must be installed to use this method.

        Parameters
        ----------
        colorscale : str, optional
            A colorscale name understood by plotly.

        min_score : float, optional
            Lines for models that have a score lower than ``min_score`` are not
            displayed, and the color scale lower bound is adjusted accordingly.
            This is useful when we are only interested in the models that
            perform well, to make the plot less cluttered and to make better
            use of the colorscale's range.

        Returns
        -------
        Plotly Figure
        """
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


class _XyParamSearch(_XyPipelineMixin, ParamSearch):
    # Similar to _XyPipeline but for ParamSearch. See _XyPipeline docstring.

    def __init__(self, data_op, search, environment):
        self.data_op = data_op
        self.search = search
        self.environment = environment

    def __skrub_to_env_learner__(self):
        new = ParamSearch(self.data_op, self.search)
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
