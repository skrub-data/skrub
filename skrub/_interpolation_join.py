import joblib
import pandas as pd
from sklearn import base, compose, ensemble

from skrub._table_vectorizer import TableVectorizer


class InterpolationJoin(base.BaseEstimator):
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

        InterpolationJoin(annual_avg_temp, on=["latitude", "longitude"]).fit_transform(
            buildings
        )

    Parameters
    ----------
    right_table : DataFrame
        The table to be joined to the argument of ``transform``.
        ``right_table`` is used to train a model that takes as inputs the
        contents of the columns listed in ``right_on``, and predicts the contents
        of the other columns. In the example above, we want our transformer to
        add temperature data to the table it is operating on. Therefore,
        ``right_table`` is the ``annual_avg_temp`` table.

    left_on : list of str, or str
        The columns in the left table used for joining. The left table is the
        argument of ``transform``, to which we add information inferred using
        ``right_table``. The column names listed in ``left_on`` will provide the
        inputs (features) of the interpolators at prediction (joining) time. In
        the example above, ``left_on`` is ``["latitude", "longitude"]``, which
        refer to columns in the ``buildings`` table. When joining on a single
        column, we can pass its name rather than a list: ``"latitude"`` is
        equivalent to ``["latitude"]``.

    right_on : list of str, or str
        The columns in ``right_table`` used for joining. Their number and types
        must match those of the `left_on` columns in the left table. These
        columns provide the features for the estimators to be fitted. As for
        ``left_on``, it is possible to pass a string when using a single column.

    on : list of str, or str
        Column names to use both `left_on` and `right_on`, when they are the
        same. Provide either `on` (only) or both `left_on` and `right_on`.

    suffix : str
        Suffix to append to the ``right_table``'s column names. You can use it
        to avoid duplicate column names in the join.

    regressor : scikit-learn regressor
        Model used to predict the numerical columns of ``right_table``.

    classifier : scikit-learn classifier
        Model used to predict the categorical (string) columns of
        ``right_table``.

    vectorizer : scikit-learn transformer that can operate on a DataFrame or None
        If provided, it is used to transform the feature columns before passing
        them to the scikit-learn estimators. This is useful if we are joining
        on columns that cannot be used directly, such as timestamps or strings
        representing high-cardinality categories. If None, no transformation is
        applied.

    n_jobs : int
        Number of joblib workers to use Depending on the estimators used and
        the contents of ``right_table``, several estimators may need to be
        fitted -- for example one for continuous outputs (regressor) and one
        for categorical outputs (classifier), or one for each column when the
        provided estimators do not support multi-output tasks. Fitting and
        querying these estimators can be done in parallel.

    Attributes
    ----------
    vectorizer_ : scikit-learn transformer
        The transformer used to vectorize the feature columns.

    estimators_ : list of dicts
        The estimators used to infer values to be joined. Each entry in this
        list is a dictionary with keys ``"estimator"`` (the fitted estimator) and
        ``"columns"`` (the list of columns that it is trained to predict).

    See Also
    --------
    Joiner :
        Works in a similar way but instead of inferring values, picks the
        closest row from the right table.

    Examples
    --------
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

    >>> InterpolationJoin(
    ...     annual_avg_temp,
    ...     on=["latitude", "longitude"],
    ...     regressor=KNeighborsRegressor(2),
    ... ).fit_transform(buildings)
       latitude  longitude  n_stories  avg_temp
    0       1.0        1.0          3      10.5
    1       2.0        2.0          7      15.5
    """

    def __init__(
        self,
        right_table,
        *,
        left_on=None,
        right_on=None,
        on=None,
        suffix="",
        regressor=ensemble.HistGradientBoostingRegressor(),
        classifier=ensemble.HistGradientBoostingClassifier(),
        vectorizer=TableVectorizer(),
        n_jobs=1,
    ):
        self.right_table = right_table
        self.left_on = left_on
        self.right_on = right_on
        self.on = on
        self.suffix = suffix
        self.regressor = regressor
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.n_jobs = n_jobs

    def fit(self, data=None, targets=None):
        """Fit estimators to the `right_table` provided during initialization.

        `data` and `targets` are for scikit-learn compatibility and they are
        ignored.

        Parameters
        ----------
        data : array-like
            Ignored.
        targets : array-like
            Ignored.

        Returns
        -------
        self : InterpolationJoin
            Returns self.
        """
        del data, targets
        self._check_inputs()
        X_values = self.vectorizer_.fit_transform(
            self.right_table.loc[:, self._right_on]
        )
        estimators = self._get_estimator_assignments()
        self.estimators_ = joblib.Parallel(self.n_jobs, verbose=3)(
            joblib.delayed(_fit)(
                X_values,
                self.right_table,
                assignment["columns"],
                assignment["estimator"],
            )
            for assignment in estimators
        )
        return self

    def _check_inputs(self):
        if self.vectorizer is None:
            self.vectorizer_ = compose.ColumnTransformer([], remainder="passthrough")
        else:
            self.vectorizer_ = base.clone(self.vectorizer)
        self._check_condition()

    def _check_condition(self):
        if self.on is not None:
            if self.right_on is not None or self.left_on is not None:
                raise ValueError(
                     "Can only pass argument 'on' OR 'left_on' and "
                     "'right_on', not a combination of both."
                 )
            left_on, right_on = self.on, self.on
        else:
            if self.right_on is None or self.left_on is None:
                raise ValueError("Must pass EITHER 'on', OR ('left_on' AND 'right_on').")
            left_on, right_on = self.left_on, self.right_on
        self._left_on = [left_on] if isinstance(left_on, str) else list(left_on)
        self._right_on = [right_on] if isinstance(right_on, str) else list(right_on)

    def transform(self, left_table):
        """Transform a table by joining inferred values to it.

        The values of the `left_on` columns in `left_table` are used to predict
        likely values for the contents of a matching row in `self.right_table`.

        Parameters
        ----------
        left_table : DataFrame
            The table to transform.

        Returns
        -------
        join : DataFrame
            The result of the join between `left_table` and inferred rows from
            ``self.right_table``.
        """
        X_values = self.vectorizer_.transform(left_table.loc[:, self._left_on])
        interpolated_parts = joblib.Parallel(self.n_jobs, verbose=3)(
            joblib.delayed(_predict)(
                X_values, assignment["columns"], assignment["estimator"]
            )
            for assignment in self.estimators_
        )
        interpolated_parts = _add_column_name_suffix(interpolated_parts, self.suffix)
        original_index = left_table.index
        return pd.concat(
            [left_table.reset_index(drop=True)] + interpolated_parts, axis=1
        ).set_index(original_index)

    def fit_transform(self, left_table):
        """Fit the estimators and perform the join.

        Parameters
        ----------
        left_table : DataFrame
            The table to transform.

        Returns
        -------
        join : DataFrame
            The result of the join between `left_table` and inferred rows from
            ``self.right_table``.
        """
        return self.fit().transform(left_table)

    def _get_estimator_assignments(self):
        """Identify column groups to be predicted together and assign them an estimator.

        In many cases, a single estimator cannot handle all the target columns.
        This function groups columns that can be handled together and returns a
        list of dictionaries, each with keys "columns" and "estimator".

        Regression and classification targets are always handled separately.

        Any column with missing values is handled separately from the rest,
        because missing values in Y have to be dropped and the corresponding
        rows may have valid values in the other columns.

        When the estimator does not handle multi-output, an estimator is fitted
        separately to each column.

        """
        right_table = self.right_table.drop(self._right_on, axis=1)
        assignments = []
        regression_columns = right_table.select_dtypes("number")
        assignments.extend(
            _get_assignments_for_estimator(regression_columns, self.regressor)
        )
        classification_columns = right_table.select_dtypes(
            ["object", "string", "category"]
        )
        assignments.extend(
            _get_assignments_for_estimator(classification_columns, self.classifier)
        )
        return assignments


def _get_assignments_for_estimator(table, estimator):
    """Get the groups of columns assigned to a single estimator.

    (which is either the regressor or the classifier)."""
    if not table.shape[1]:
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
    return getattr(estimator, "_get_tags")().get("multioutput", False)


def _fit(X_values, right_table, target_columns, estimator):
    estimator = base.clone(estimator)
    kept_rows = right_table.loc[:, target_columns].notnull().all(axis=1).to_numpy()
    X_values = X_values[kept_rows]
    right_table = right_table[kept_rows]
    Y = right_table.loc[:, target_columns]
    Y_values = Y.to_numpy()
    if Y_values.shape[-1] == 1:
        Y_values = Y_values.ravel()
    estimator.fit(X_values, Y_values)
    return {"columns": Y.columns, "estimator": estimator}


def _predict(X_values, columns, estimator):
    Y_values = estimator.predict(X_values)
    return pd.DataFrame(data=Y_values, columns=columns)


def _add_column_name_suffix(dataframes, suffix):
    if suffix == "":
        return dataframes
    renamed = []
    for df in dataframes:
        renamed.append(df.rename(columns={c: f"{c}{suffix}" for c in df.columns}))
    return renamed
