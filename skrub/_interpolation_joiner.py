import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.utils._tags import _safe_tags

from skrub import _join_utils, _utils
from skrub._minhash_encoder import MinHashEncoder
from skrub._table_vectorizer import TableVectorizer

DEFAULT_VECTORIZER = TableVectorizer(high_cardinality_transformer=MinHashEncoder())
DEFAULT_REGRESSOR = HistGradientBoostingRegressor()
DEFAULT_CLASSIFIER = HistGradientBoostingClassifier()


class InterpolationJoiner(TransformerMixin, BaseEstimator):
    """Join with a table augmented by machine-learning predictions.

    This is similar to a usual equi-join, but instead of looking for actual
    rows in the right table that satisfy the join condition, we estimate what
    those rows would contain if they existed in the table.

    Suppose we want to join a table ``buildings(latitude, longitude, n_stories)``
    with a table ``annual_avg_temp(latitude, longitude, avg_temp)``. Our annual
    average temperature table may not contain data for the exact latitude and
    longitude of our buildings. However, we can interpolate what we need from
    the data points it does contain. Using ``annual_avg_temp``, we train a
    model to predict the temperature, given the latitude and longitude. Then,
    we use this model to estimate the values we want to add to our
    ``buildings`` table. In a way we are joining ``buildings`` to a virtual
    table, in which rows for any (latitude, longitude) location are inferred,
    rather than retrieved, when requested. This is done with::

        InterpolationJoiner(
            annual_avg_temp, on=["latitude", "longitude"]
        ).fit_transform(buildings)

    Parameters
    ----------
    aux_table : DataFrame
        The (auxiliary) table to be joined to the `main_table` (which is the
        argument of ``transform``). ``aux_table`` is used to train a model that
        takes as inputs the contents of the columns listed in ``aux_key``, and
        predicts the contents of the other columns. In the example above, we
        want our transformer to add temperature data to the table it is
        operating on. Therefore, ``aux_table`` is the ``annual_avg_temp``
        table.

    main_key : list of str, or str
        The columns in the main table used for joining. The main table is the
        argument of ``transform``, to which we add information inferred using
        ``aux_table``. The column names listed in ``main_key`` will provide the
        inputs (features) of the interpolators at prediction (joining) time. In
        the example above, ``main_key`` is ``["latitude", "longitude"]``, which
        refer to columns in the ``buildings`` table. When joining on a single
        column, we can pass its name rather than a list: ``"latitude"`` is
        equivalent to ``["latitude"]``.

    aux_key : list of str, or str
        The columns in ``aux_table`` used for joining. Their number and types
        must match those of the ``main_key`` columns in the main table. These
        columns provide the features for the estimators to be fitted. As for
        ``main_key``, it is possible to pass a string when using a single
        column.

    key : list of str, or str
        Column names to use for both `main_key` and `aux_key`, when they are
        the same. Provide either `key` (only) or both `main_key` and `aux_key`.

    suffix : str
        Suffix to append to the ``aux_table``'s column names. You can use it
        to avoid duplicate column names in the join.

    regressor : scikit-learn regressor
        Model used to predict the numerical columns of ``aux_table``.

    classifier : scikit-learn classifier
        Model used to predict the categorical (string) columns of ``aux_table``.

    vectorizer : scikit-learn transformer that can operate on a DataFrame
        Used to transform the feature columns before passing them to the
        scikit-learn estimators. This is useful if we are joining on columns
        that need some transformation, such as dates or strings representing
        high-cardinality categories. By default we use a ``MinHashEncoder`` to
        vectorize text columns. This is because the ``MinHashEncoder`` is very
        fast and usually gives good results with downstream learners based on
        trees like the gradient-boosted trees used by default for ``regressor``
        and ``classifier``. If you replace the default regressor and classifier
        with models such as nearest-neighbors or linear models, consider
        passing ``vectorizer=TableVectorizer()`` which will encode text with a
        ``GapEncoder`` rather than a ``MinHashEncoder``.

    n_jobs : int or None
        Number of jobs to run in parallel. ``None`` means 1 unless in a
        ``joblib.parallel_backend`` context. -1 means using all processors.
        Depending on the estimators used and the contents of ``aux_table``,
        several estimators may need to be fitted -- for example one for
        continuous outputs (regressor) and one for categorical outputs
        (classifier), or one for each column when the provided estimators do
        not support multi-output tasks. Fitting and querying these estimators
        can be done in parallel.

    on_estimator_failure : "warn", "raise" or "pass"
        How to handle exceptions raised when fitting one of the estimators
        (regressors and classifiers) or querying them for a prediction. If
        "raise", exceptions are propagated. If "pass" (i) if an exception is
        raised during ``fit`` the corresponding columns are ignored -- they
        will not appear in the join and (ii) if an exception is raised during
        ``transform``, the corresponding column will be filled with nulls.
        Columns are filled with nulls during ``transform`` rather than dropped
        so that the output always has the same shape. If "warn" (the default),
        behave like "pass" but issue a warning.

    Attributes
    ----------
    vectorizer_ : scikit-learn transformer
        The transformer used to vectorize the feature columns.

    estimators_ : list of dicts
        The estimators used to infer values to be joined. Each entry in this
        list is a dictionary with keys ``"estimator"`` (the fitted estimator)
        and ``"columns"`` (the list of columns in ``aux_table`` that it is
        trained to predict).

    See Also
    --------
    Joiner :
        Works in a similar way but instead of inferring values, picks the
        closest row from the auxiliary table.

    Examples
    --------
    >>> buildings = pd.DataFrame(
    ...     {"latitude": [1.0, 2.0], "longitude": [1.0, 2.0], "n_stories": [3, 7]}
    ... )
    >>> annual_avg_temp = pd.DataFrame(
    ...     {
    ...         "latitude": [1.2, 0.9, 1.9, 1.7, 5.0],
    ...         "longitude": [0.8, 1.1, 1.8, 1.8, 5.0],
    ...         "avg_temp": [10.0, 11.0, 15.0, 16.0, 20.0],
    ...     }
    ... )

    >>> buildings
       latitude  longitude  n_stories
    0       1.0        1.0          3
    1       2.0        2.0          7

    >>> annual_avg_temp
       latitude  longitude  avg_temp
    0       1.2        0.8      10.0
    1       0.9        1.1      11.0
    2       1.9        1.8      15.0
    3       1.7        1.8      16.0
    4       5.0        5.0      20.0

    >>> from sklearn.neighbors import KNeighborsRegressor

    >>> InterpolationJoiner(
    ...     annual_avg_temp,
    ...     key=["latitude", "longitude"],
    ...     regressor=KNeighborsRegressor(2),
    ... ).fit_transform(buildings)
       latitude  longitude  n_stories  avg_temp
    0       1.0        1.0          3      10.5
    1       2.0        2.0          7      15.5
    """

    def __init__(
        self,
        aux_table,
        *,
        main_key=None,
        aux_key=None,
        key=None,
        suffix="",
        regressor=DEFAULT_REGRESSOR,
        classifier=DEFAULT_CLASSIFIER,
        vectorizer=DEFAULT_VECTORIZER,
        n_jobs=None,
        on_estimator_failure="warn",
    ):
        self.aux_table = aux_table
        self.main_key = main_key
        self.aux_key = aux_key
        self.key = key
        self.suffix = suffix
        self.regressor = _utils.clone_if_default(regressor, DEFAULT_REGRESSOR)
        self.classifier = _utils.clone_if_default(classifier, DEFAULT_CLASSIFIER)
        self.vectorizer = _utils.clone_if_default(vectorizer, DEFAULT_VECTORIZER)
        self.n_jobs = n_jobs
        self.on_estimator_failure = on_estimator_failure

    def fit(self, X, y=None):
        """Fit estimators to the `aux_table` provided during initialization.

        `X` and `y` are mostly for scikit-learn compatibility.

        Parameters
        ----------
        X : array-like or None
            The main table to which ``self.aux_table`` could be joined. If `X`
            is not ``None``, an error is raised if any of the matching columns
            listed in ``self.main_key`` (or ``self.key``) is missing from `X`.

        y : array-like
            Ignored; only exists for compatibility with scikit-learn.

        Returns
        -------
        self : InterpolationJoiner
            Returns self.
        """
        del y
        self._check_inputs()
        if X is not None:
            _join_utils.check_missing_columns(X, self._main_key, "'X' (the main table)")
        key_values = self.vectorizer_.fit_transform(self.aux_table[self._aux_key])
        estimators = self._get_estimator_assignments()
        fit_results = joblib.Parallel(self.n_jobs)(
            joblib.delayed(_fit)(
                key_values,
                self.aux_table[assignment["columns"]],
                assignment["estimator"],
                propagate_exceptions=(self.on_estimator_failure == "raise"),
            )
            for assignment in estimators
        )
        fit_results = self._check_fit_results(fit_results)
        for res in fit_results:
            del res["failed"]
        self.estimators_ = fit_results
        return self

    def _check_inputs(self):
        self.vectorizer_ = clone(self.vectorizer)
        self.classifier_ = clone(self.classifier)
        self.regressor_ = clone(self.regressor)
        self._main_key, self._aux_key = _join_utils.check_key(
            self.main_key, self.aux_key, self.key
        )
        _join_utils.check_missing_columns(self.aux_table, self._aux_key, "'aux_table'")

    def _check_fit_results(self, results):
        successful_results = [res for res in results if not res["failed"]]
        if self.on_estimator_failure == "pass":
            return successful_results
        failed_columns = []
        for res in results:
            if res["failed"]:
                failed_columns.extend(res["columns"])
        if not failed_columns:
            return successful_results
        warnings.warn(
            "Estimators failed to be fitted for the following"
            f" columns:\n{failed_columns}"
        )
        return successful_results

    def transform(self, X):
        """Transform a table by joining inferred values to it.

        The values of the `main_key` columns in `X` (the main table) are used
        to predict likely values for the contents of a matching row in
        `self.aux_table` (the auxiliary table).

        Parameters
        ----------
        X : DataFrame
            The (main) table to transform.

        Returns
        -------
        join : DataFrame
            The result of the join between `X` and inferred rows from
            ``self.aux_table``.
        """
        main_table = X
        _join_utils.check_missing_columns(
            main_table, self._main_key, "'X' (the main table)"
        )
        key_values = self.vectorizer_.transform(
            main_table[self._main_key].set_axis(self._aux_key, axis="columns")
        )
        prediction_results = joblib.Parallel(self.n_jobs)(
            joblib.delayed(_predict)(
                key_values,
                assignment["columns"],
                assignment["estimator"],
                propagate_exceptions=(self.on_estimator_failure == "raise"),
            )
            for assignment in self.estimators_
        )
        prediction_results = self._check_prediction_results(prediction_results)
        predictions = [res["predictions"] for res in prediction_results]
        predictions = [
            _join_utils.add_column_name_suffix(df, self.suffix) for df in predictions
        ]
        for part in predictions:
            part.index = main_table.index
        return pd.concat([main_table] + predictions, axis=1)

    def _check_prediction_results(self, results):
        checked_results = []
        failed_columns = []
        for res in results:
            new_res = dict(**res)
            if res["failed"]:
                if set(res["columns"]).issubset(
                    self.aux_table.select_dtypes("number").columns.values
                ):
                    dtype = float
                else:
                    dtype = object
                pred = pd.DataFrame(
                    columns=res["columns"],
                    index=np.arange(res["shape"][0]),
                    dtype=dtype,
                )
                new_res["predictions"] = pred
                failed_columns.extend(res["columns"])
            checked_results.append(new_res)
        if not failed_columns:
            return checked_results
        if self.on_estimator_failure == "pass":
            return checked_results
        warnings.warn(
            "Prediction failed for the following columns; output will be filled with"
            f" nulls:\n{failed_columns}"
        )
        return checked_results

    def _get_estimator_assignments(self):
        """Identify column groups to be predicted together and assign them an estimator.

        In many cases, a single estimator cannot handle all the target columns.
        This function groups columns that can be handled together and returns a
        list of dictionaries, each with keys "columns" and "estimator".

        Regression and classification targets are always handled separately.

        Any column with missing values is handled separately from the rest.
        This is due to the fact that missing values in the columns we are
        trying to predict have to be dropped, and the corresponding rows may
        have valid values in the other columns.

        When the estimator does not handle multi-output, an estimator is fitted
        separately to each column.
        """
        aux_table = self.aux_table.drop(self._aux_key, axis=1)
        assignments = []
        regression_table = aux_table.select_dtypes("number")
        assignments.extend(
            _get_assignments_for_estimator(regression_table, self.regressor_)
        )
        classification_table = aux_table.select_dtypes(["object", "string", "category"])
        assignments.extend(
            _get_assignments_for_estimator(classification_table, self.classifier_)
        )
        return assignments


def _get_assignments_for_estimator(table, estimator):
    """Get the groups of columns assigned to a single estimator.

    (which is either the regressor or the classifier)."""

    # If the complete set of columns that have to be predicted with this
    # estimator is empty (eg the estimator is the regressor and there are no
    # numerical columns), return an empty list -- no columns are assigned to
    # that estimator.
    if table.empty:
        return []
    if not _handles_multioutput(estimator):
        return [{"columns": [col], "estimator": estimator} for col in table.columns]
    columns_with_nulls = table.columns[table.isnull().any()]
    assignments = [
        {"columns": [col], "estimator": estimator} for col in columns_with_nulls
    ]
    columns_without_nulls = list(set(table.columns).difference(columns_with_nulls))
    if columns_without_nulls:
        assignments.append({"columns": columns_without_nulls, "estimator": estimator})
    return assignments


def _handles_multioutput(estimator):
    return _safe_tags(estimator).get("multioutput", False)


def _fit(key_values, target_table, estimator, propagate_exceptions):
    estimator = clone(estimator)
    kept_rows = target_table.notnull().all(axis=1).to_numpy()
    key_values = key_values[kept_rows]
    Y = target_table.to_numpy()[kept_rows]

    # Estimators that expect a single output issue a DataConversionWarning if
    # passing a column vector rather than a 1-D array
    if len(target_table.columns) == 1:
        Y = Y.ravel()
    failed = False
    try:
        estimator.fit(key_values, Y)
    except Exception:
        if propagate_exceptions:
            raise
        failed = True
        estimator = None
    return {"columns": target_table.columns, "estimator": estimator, "failed": failed}


def _predict(key_values, columns, estimator, propagate_exceptions):
    failed = False
    try:
        Y_values = estimator.predict(key_values)
    except Exception:
        if propagate_exceptions:
            raise
        failed = True
    if failed:
        predictions = None
    else:
        predictions = pd.DataFrame(data=Y_values, columns=columns)
    return {
        "predictions": predictions,
        "failed": failed,
        "columns": columns,
        "shape": (key_values.shape[0], len(columns)),
    }
