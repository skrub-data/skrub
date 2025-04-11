import pandas as pd
from sklearn import model_selection
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .. import _join_utils
from ._choosing import Choice, get_default
from ._evaluation import (
    choices,
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
from ._expressions import Apply
from ._parallel_coord import DEFAULT_COLORSCALE, plot_parallel_coord
from ._utils import X_NAME, Y_NAME, attribute_error

_FITTING_METHODS = ["fit", "fit_transform"]
_SEARCH_FITTED_ATTRIBUTES = [
    "cv_results_",
    "best_estimator_",
    "best_score_",
    "best_params_",
    "best_index_",
    "scorer_",
    "n_splits_",
    "refit_time_",
    "multimetric_",
]


def _default_sklearn_tags():
    class _DummyTransformer(TransformerMixin, BaseEstimator):
        pass

    return _DummyTransformer().__sklearn_tags__()


class _SharedDict(dict):
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


class ExprEstimator(BaseEstimator):
    """Estimator that evaluates a skrub expression.

    This class is not meant to be instantiated manually, ``ExprEstimator``
    objects are created by calling :meth:`Expr.skb.get_estimator()` on an
    expression.
    """

    def __init__(self, expr):
        self.expr = expr

    def __skrub_to_Xy_estimator__(self, environment):
        new = _XyExprEstimator(self.expr, _SharedDict(environment))
        _copy_attr(self, new, ["_is_fitted"])
        return new

    def _set_is_fitted(self, mode):
        if mode in _FITTING_METHODS:
            self._is_fitted = True

    def __sklearn_is_fitted__(self):
        return getattr(self, "_is_fitted", False)

    def fit(self, environment):
        _ = self.fit_transform(environment)
        return self

    def _eval_in_mode(self, mode, environment):
        if mode not in _FITTING_METHODS:
            check_is_fitted(self)
        result = evaluate(self.expr, mode, environment, clear=True)
        self._set_is_fitted(mode)
        return result

    def report(self, *, environment, mode, **full_report_kwargs):
        from ._inspection import full_report

        if mode not in _FITTING_METHODS:
            check_is_fitted(self)

        full_report_kwargs["clear"] = True
        result = full_report(
            self.expr, environment=environment, mode=mode, **full_report_kwargs
        )
        if mode == "fit" and result["result"] is not None:
            result["result"] = self
        self._set_is_fitted(mode)
        return result

    def __getattr__(self, name):
        if name not in supported_modes(self.expr):
            attribute_error(self, name)

        def f(*args, **kwargs):
            return self._eval_in_mode(name, *args, **kwargs)

        f.__name__ = name
        return f

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        if not deep:
            return params
        params.update({f"expr__{k}": v for k, v in get_params(self.expr).items()})
        return params

    def set_params(self, **params):
        if "expr" in params:
            self.expr = params.pop("expr")
        set_params(self.expr, {int(k.lstrip("expr__")): v for k, v in params.items()})
        return self

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
            may be wrapped in a ``skrub.OnEachColumn`` or ``skrub.OnSubFrame``,
            see examples below.

        Examples
        --------
        >>> from sklearn.decomposition import PCA
        >>> from sklearn.dummy import DummyClassifier
        >>> import skrub
        >>> from skrub import selectors as s

        >>> orders = skrub.toy_orders()
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
        >>> estimator = pred.skb.get_estimator()
        >>> estimator.fit({'X': orders.X, 'y': orders.y})
        ExprEstimator(expr=<classifier | Apply DummyClassifier>)

        We can retrieve the fitted transformer for a given step with
        ``find_fitted_estimator``:

        >>> estimator.find_fitted_estimator("classifier")
        DummyClassifier()

        Depending on the parameters passed to ``skb.apply()``, the estimator we provide
        can be wrapped in a skrub transformer that applies it to several columns in the
        input, or to a subset of the columns in a dataframe. In other cases it may be
        applied without any wrapping. We provide examples for those 3 different cases
        below.

        Case 1: the ``StringEncoder`` is a skrub single-column transformer: it
        transforms a single column. In the pipeline it gets wrapped in a
        ``skrub.OnEachColumn`` which independently fits a separate instance of the
        ``StringEncoder`` to each of the columns it transforms (in this case there is
        only one column, ``'product'``). The individual transformers can be found in the
        fitted attribute ``transformers_`` which maps column names to the corresponding
        fitted transformer.

        >>> encoder = estimator.find_fitted_estimator('product_encoder')
        >>> encoder.transformers_
        {'product': StringEncoder(n_components=2)}
        >>> encoder.transformers_['product'].vectorizer_.vocabulary_
        {' pe': 2, 'pen': 12, 'en ': 8, ' pen': 3, 'pen ': 13, ' cu': 0, 'cup': 6, 'up ': 18, ' cup': 1, 'cup ': 7, ' sp': 4, 'spo': 16, 'poo': 14, 'oon': 10, 'on ': 9, ' spo': 5, 'spoo': 17, 'poon': 15, 'oon ': 11}

        This case (wrapping in ``OnEachColumn``) happens when the estimator is a skrub
        single-column transformer (it has a ``__single_column_transformer__``
        attribute), we pass ``.skb.apply(how='columnwise')`` or we pass
        ``.skb.apply(allow_reject=True)``.

        Case 2: the ``PCA`` is a regular scikit-learn transformer. In the pipeline it
        gets wrapped in a ``skrub.OnSubFrame`` which applies it to the subset of columns
        in the dataframe selected by the ``cols`` argument passed to ``.skb.apply()``.
        The fitted ``PCA`` can be found in the fitted attribute ``transformer_``.

        >>> pca = estimator.find_fitted_estimator('pca')
        >>> pca
        OnSubFrame(cols=glob('date_*'), transformer=PCA(n_components=2))
        >>> pca.transformer_
        PCA(n_components=2)
        >>> pca.transformer_.mean_
        array([2020.,    4.,    4.], dtype=float32)

        This case (wrapping in ``OnSubFrame``) happens when the estimator is a
        scikit-learn transformer but not a single-column transformer.

        The ``DummyRegressor`` is a scikit-learn predictor. In the pipeline it gets
        applied directly to the input dataframe without any wrapping.

        >>> classifier = estimator.find_fitted_estimator('classifier')
        >>> classifier
        DummyClassifier()
        >>> classifier.class_prior_
        array([0.75, 0.25])

        This case (no wrapping) happens when the estimator is a scikit-learn predictor
        (not a transformer), the input is not a dataframe (e.g. it is a numpy array), or
        we pass ``.skb.apply(how='full_frame')``.
        """  # noqa: E501
        node = find_node_by_name(self.expr, name)
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
                f"Node {name!r} has not been fitted. Call fit() on the estimator "
                "before attempting to retrieve fitted sub-estimators."
            )
        return node._skrub_impl.estimator_

    def sub_estimator(self, name):
        """Extract the part of the pipeline that leads up to the given step.

        This is similar to slicing a scikit-learn pipeline. It can be useful
        for example to drive the hyperparameter selection with a supervised
        task but then extract only the part of the pipeline that performs
        feature extraction.

        The target step must have been given a name with ``.skb.set_name()``.

        Parameters
        ----------
        name : str
            The name of the intermediate step we want to extract.

        Returns
        -------
        ExprEstimator
            An estimator that performs all the transformations leading up to
            (and including) the required step.

        Examples
        --------
        >>> from sklearn.dummy import DummyClassifier
        >>> import skrub

        >>> orders = skrub.toy_orders()
        >>> X, y = skrub.X(), skrub.y()
        >>> pred = (
        ...     X.skb.apply(
        ...         skrub.TableVectorizer(datetime=skrub.DatetimeEncoder(add_total_seconds=False))
        ...     )
        ...     .skb.set_name("vectorizer")
        ...     .skb.apply(DummyClassifier(), y=y)
        ... )
        >>> estimator = pred.skb.get_estimator()
        >>> estimator.fit({"X": orders.X, "y": orders.y})
        ExprEstimator(expr=<Apply DummyClassifier>)
        >>> estimator.predict({"X": orders.X})
        array([False, False, False, False])

        Truncate the pipeline after vectorization:

        >>> vectorizer = estimator.sub_estimator("vectorizer")
        >>> vectorizer
        ExprEstimator(expr=<vectorizer | Apply TableVectorizer>)
        >>> vectorizer.transform({"X": orders.X})
            ID  product_cup  product_pen  ...  date_year  date_month  date_day
        0  1.0          0.0          1.0  ...     2020.0         4.0       3.0
        1  2.0          1.0          0.0  ...     2020.0         4.0       4.0
        2  3.0          1.0          0.0  ...     2020.0         4.0       4.0
        3  4.0          0.0          0.0  ...     2020.0         4.0       5.0

        Note this differs from ``find_fitted_estimator`` which extracts the inner
        scikit-learn estimator that has been fitted inside of a single step.

        This contains the full transformation up to the given step:

        >>> estimator.sub_estimator("vectorizer")
        ExprEstimator(expr=<vectorizer | Apply TableVectorizer>)

        This contains only the inner ``TableVectorizer`` that was fitted inside of the
        ``"vectorizer"`` step:

        >>> estimator.find_fitted_estimator("vectorizer")
        OnSubFrame(transformer=TableVectorizer(datetime=DatetimeEncoder(add_total_seconds=False)))
        """  # noqa: E501
        node = find_node_by_name(self.expr, name)
        if node is None:
            raise KeyError(name)
        new = self.__class__(node)
        _copy_attr(self, new, ["_is_fitted"])
        return new


def _to_Xy_estimator(estimator, environment):
    return estimator.__skrub_to_Xy_estimator__(environment)


def _to_env_estimator(estimator):
    return estimator.__skrub_to_env_estimator__()


def _get_classes(expr):
    first = find_first_apply(expr.expr)
    if first is None:
        attribute_error(expr, "classes_")
    try:
        estimator = first._skrub_impl.estimator_
    except AttributeError:
        attribute_error(expr, "classes_")
    return estimator.classes_


class _XyEstimatorMixin:
    def _get_env(self, X, y):
        xy_environment = {X_NAME: X}
        if y is not None:
            xy_environment[Y_NAME] = y
        return {**self.environment, **xy_environment}

    @property
    def _estimator_type(self):
        first = find_first_apply(self.expr)
        if first is None:
            return "transformer"
        try:
            return get_default(first._skrub_impl.estimator)._estimator_type
        except AttributeError:
            return "transformer"

    if hasattr(BaseEstimator, "__sklearn_tags__"):
        # scikit-learn >= 1.6

        def __sklearn_tags__(self):
            first = find_first_apply(self.expr)
            if first is None:
                return _default_sklearn_tags()
            return get_default(first._skrub_impl.estimator).__sklearn_tags__()

    @property
    def classes_(self):
        try:
            return _get_classes(self.expr)
        except AttributeError:
            attribute_error(self, "classes_")


class _XyExprEstimator(_XyEstimatorMixin, ExprEstimator):
    def __init__(self, expr, environment):
        self.expr = expr
        self.environment = environment

    def __skrub_to_env_estimator__(self):
        new = ExprEstimator(self.expr)
        _copy_attr(self, new, ["_is_fitted"])
        return new

    def fit(self, X, y=None):
        _ = self.fit_transform(X, y=y)
        return self

    def _eval_in_mode(self, mode, X, y=None):
        result = evaluate(self.expr, mode, self._get_env(X, y), clear=True)
        self._set_is_fitted(mode)
        return result


def _find_Xy(expr):
    x_node = find_X(expr)
    if x_node is None:
        raise ValueError('expr should have a node marked with "mark_as_X()"')
    result = {"X": x_node}
    if (y_node := find_y(expr)) is not None:
        result["y"] = y_node
    else:
        first = find_first_apply(expr)
        if first is None:
            return result
        if getattr(first._skrub_impl, "y", None) is not None:
            # the estimator requests a y so some node must have been
            # marked as y
            raise ValueError('expr should have a node marked with "mark_as_y()"')
    return result


def _compute_Xy(expr, environment):
    Xy = _find_Xy(expr.skb.clone())
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
    else:
        y = None
    return X, y


def cross_validate(expr_estimator, environment, **cv_params):
    """Cross-validate an estimator built from an expression.

    This runs cross-validation from an estimator that was built from a skrub
    expression with ``.skb.get_estimator()``, ``.skb.get_grid_search()`` or
    ``.skb.get_randomized_search()``.

    It is useful to run nested cross-validation of a grid search or randomized
    search.

    Parameters
    ----------
    expr_estimator : estimator
        An estimator generated from a skrub expression.

    environment : dict
        Bindings for variables contained in the expression.

    cv_params : dict
        All other named arguments are forwarded to
        ``sklearn.model_selection.cross_validate``.

    Returns
    -------
    dict
        Cross-validation results.

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
    >>> search = pred.skb.get_randomized_search(random_state=0)
    >>> skrub.cross_validate(search, pred.skb.get_data())['test_score'] # doctest: +SKIP
    array([0.75, 0.95, 0.85, 0.85, 0.85])
    """
    estimator = _to_Xy_estimator(expr_estimator, environment)
    X, y = _compute_Xy(expr_estimator.expr, environment)
    result = model_selection.cross_validate(
        estimator,
        X,
        y,
        **cv_params,
    )
    if (estimators := result.get("estimator", None)) is None:
        return result
    result["estimator"] = [_to_env_estimator(e) for e in estimators]
    return result


def train_test_split(
    expr, environment, splitter=model_selection.train_test_split, **splitter_kwargs
):
    """Split an environment into a training an testing environments.

    Parameters
    ----------
    expr : skrub expression
        The expression to be evaluated in the environment.

    environment : dict
        The environment (dict mapping variable names to values) containing the
        full data.

    splitter : function, optional
        The function used to split X and y once they have been computed. By
        default, ``sklearn.train_test_split`` is used.

    splitter_kwargs
        Additional named arguments to pass to the splitter.

    Returns
    -------
    dict
        The return value is slightly different than scikit-learn's. Rather than
        a tuple, it returns a dictionary with the following keys:

        - train: a dictionary containing the training environment
        - test: a dictionary containing the test environment
        - X_train: the value of the variable marked with ``skb.mark_as_x()`` in
          the train environment
        - X_test: the value of the variable marked with ``skb.mark_as_x()`` in
          the test environment
        - y_train: the value of the variable marked with ``skb.mark_as_y()`` in
          the train environment, if there is one (may not be the case for
          unsupervised learning).
        - y_test: the value of the variable marked with ``skb.mark_as_y()`` in
          the test environment, if there is one (may not be the case for
          unsupervised learning).

    Examples
    --------
    >>> import skrub
    >>> from sklearn.dummy import DummyClassifier
    >>> from sklearn.metrics import accuracy_score

    >>> orders = skrub.var("orders", skrub.toy_orders().orders)
    >>> X = orders.skb.drop("delayed").skb.mark_as_X()
    >>> y = orders["delayed"].skb.mark_as_y()
    >>> delayed = X.skb.apply(skrub.TableVectorizer()).skb.apply(
    ...     DummyClassifier(), y=y
    ... )

    >>> split = skrub.train_test_split(delayed, delayed.skb.get_data(), random_state=0)
    >>> split.keys()
    dict_keys(['train', 'test', 'X_train', 'X_test', 'y_train', 'y_test'])
    >>> estimator = delayed.skb.get_estimator()
    >>> estimator.fit(split["train"])
    ExprEstimator(expr=<Apply DummyClassifier>)
    >>> estimator.score(split["test"])
    0.0
    >>> predictions = estimator.predict(split["test"])
    >>> accuracy_score(split["y_test"], predictions)
    0.0
    """
    X, y = _compute_Xy(expr, environment)
    if y is None:
        X_train, X_test = splitter(X, **splitter_kwargs)
    else:
        X_train, X_test, y_train, y_test = splitter(X, y, **splitter_kwargs)
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


# TODO with ParameterGrid and ParameterSampler we can generate the list of
# candidates so we can provide more than just a score, eg full predictions for
# each sampled param combination.


class ParamSearch(BaseEstimator):
    """Estimator that evaluates a skrub expression with hyperparameter tuning.

    This class is not meant to be instantiated manually, ``ParamSearch``
    objects are created by calling :meth:`Expr.skb.get_grid_search()` or
    :meth:`Expr.skb.get_randomized_search()` on an expression.
    """

    def __init__(self, expr, search):
        self.expr = expr
        self.search = search

    def __skrub_to_Xy_estimator__(self, environment):
        new = _XyParamSearch(self.expr, self.search, _SharedDict(environment))
        _copy_attr(self, new, _SEARCH_FITTED_ATTRIBUTES)
        return new

    def _get_param_grid(self):
        grid = param_grid(self.expr)
        new_grid = []
        for subgrid in grid:
            subgrid = {f"expr__{k}": v for k, v in subgrid.items()}
            new_grid.append(subgrid)
        return new_grid

    def fit(self, environment):
        estimator = _XyExprEstimator(self.expr, _SharedDict(environment))
        search = clone(self.search)
        search.estimator = estimator
        param_grid = self._get_param_grid()
        if hasattr(search, "param_grid"):
            search.param_grid = param_grid
        else:
            assert hasattr(search, "param_distributions")
            search.param_distributions = param_grid
        X, y = _compute_Xy(self.expr, environment)
        search.fit(X, y)
        _copy_attr(search, self, _SEARCH_FITTED_ATTRIBUTES)
        try:
            self.best_estimator_ = _to_env_estimator(search.best_estimator_)
        except AttributeError:
            # refit is set to False, there is no best_estimator_
            pass
        return self

    def __getattr__(self, name):
        if name not in supported_modes(self.expr):
            attribute_error(self, name)

        def f(*args, **kwargs):
            return self._call_predictor_method(name, *args, **kwargs)

        f.__name__ = name
        return f

    def _call_predictor_method(self, name, environment):
        check_is_fitted(self, "cv_results_")
        if not hasattr(self, "best_estimator_"):
            raise AttributeError(
                "This parameter search was initialized with `refit=False`. "
                f"{name} is available only after refitting on the best parameters. "
                "Please pass another value to `refit` or fit an estimator manually "
                "using the `best_params_` or `cv_results_` attributes."
            )
        return getattr(self.best_estimator_, name)(environment)

    @property
    def results_(self):
        """
        Cross-validation results containing parameters and scores in a dataframe.
        """
        try:
            return self._get_cv_results_table()
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
            return self._get_cv_results_table(detailed=True)
        except NotFittedError:
            attribute_error(self, "results_")

    def _get_cv_results_table(self, return_metadata=False, detailed=False):
        check_is_fitted(self, "cv_results_")
        expr_choices = choices(self.expr)

        all_rows = []
        log_scale_columns = set()
        for params in self.cv_results_["params"]:
            row = {}
            for param_id, param in params.items():
                choice = expr_choices[int(param_id.lstrip("expr__"))]
                if isinstance(choice, Choice):
                    if choice.outcome_names is not None:
                        value = choice.outcome_names[param]
                    else:
                        value = choice.outcomes[param]
                else:
                    value = param
                    if choice.log:
                        log_scale_columns.add(choice.name)
                row[choice.name] = value
            all_rows.append(row)

        metadata = {"log_scale_columns": list(log_scale_columns)}
        table = pd.DataFrame(all_rows)
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
        metadata["log_scale_columns"] = [
            renaming[c] for c in metadata["log_scale_columns"]
        ]
        for k in result_keys[: len(metric_names)][::-1]:
            table.insert(0, k, self.cv_results_[k])
        if detailed:
            for k in result_keys[len(metric_names) :]:
                if k in self.cv_results_:
                    table.insert(table.shape[1], k, self.cv_results_[k])
        table = table.sort_values(
            list(table.columns)[0], ascending=False, ignore_index=True, kind="stable"
        )
        return (table, metadata) if return_metadata else table

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
        if not cv_results.shape[0]:
            raise ValueError("No results to plot")
        return plot_parallel_coord(cv_results, metadata, colorscale=colorscale)


class _XyParamSearch(_XyEstimatorMixin, ParamSearch):
    def __init__(self, expr, search, environment):
        self.expr = expr
        self.search = search
        self.environment = environment

    def __skrub_to_env_estimator__(self):
        new = ParamSearch(self.expr, self.search)
        _copy_attr(self, new, _SEARCH_FITTED_ATTRIBUTES)
        return new

    @property
    def classes_(self):
        if not hasattr(self, "best_estimator_"):
            attribute_error(self, "classes_")
        try:
            return _get_classes(self.best_estimator_.expr)
        except AttributeError:
            attribute_error(self, "classes_")

    def fit(self, X, y=None):
        super().fit(self._get_env(X, y))
        return self

    def _call_predictor_method(self, name, X, y=None):
        return getattr(self.best_estimator_, name)(self._get_env(X, y))
