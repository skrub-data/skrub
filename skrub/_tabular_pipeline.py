import warnings

import sklearn
from sklearn import ensemble
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.fixes import parse_version

from ._datetime_encoder import DatetimeEncoder
from ._sklearn_compat import get_tags
from ._string_encoder import StringEncoder
from ._table_vectorizer import TableVectorizer
from ._to_categorical import ToCategorical

_HGBT_CLASSES = (
    ensemble.HistGradientBoostingClassifier,
    ensemble.HistGradientBoostingRegressor,
)
_TREE_ENSEMBLE_CLASSES = (
    ensemble.HistGradientBoostingClassifier,
    ensemble.HistGradientBoostingRegressor,
    ensemble.RandomForestClassifier,
    ensemble.RandomForestRegressor,
)


def tabular_learner(estimator, *, n_jobs=None):
    """Get a simple machine-learning pipeline for tabular data.

    .. deprecated:: 0.6.0
        The functionality provided by this function is now implemented in
        :func:`~skrub.tabular_pipeline`.

    ``'regressor'``, ``'regression'``, ``'classifier'``, ``'classification'``, this
    function creates a scikit-learn pipeline that extracts numeric features, imputes
    missing values and scales the data if necessary, then applies the estimator.

    .. note::
       The heuristics used by the ``tabular_pipeline``
       to define an appropriate preprocessing based on the ``estimator`` may change
       in future releases.

    .. versionchanged:: 0.6.0
        The high cardinality encoder has been changed from
        :class:`~skrub.MinHashEncoder` to :class:`~skrub.StringEncoder`.

    Parameters
    ----------
    estimator : {"regressor", "regression", "classifier", "classification"} or sklearn.base.BaseEstimator
        The estimator to use as the final step in the pipeline. Based on the type of
        estimator, the previous preprocessing steps and their respective parameters are
        chosen. The possible values are:

        - ``'regressor'`` or ``'regression'``: a
          :obj:`~sklearn.ensemble.HistGradientBoostingRegressor` is used as the final
          step;
        - ``'classifier'`` or ``'classification'``: a
          :obj:`~sklearn.ensemble.HistGradientBoostingClassifier` is used as the final
          step;
        - a scikit-learn estimator: the provided estimator is used as the final step.

    n_jobs : int, default=None
        Number of jobs to run in parallel in the :obj:`TableVectorizer` step. ``None``
        means 1 unless in a joblib ``parallel_backend`` context. ``-1`` means using all
        processors.

    Returns
    -------
    Pipeline
        A scikit-learn :obj:`~sklearn.pipeline.Pipeline` chaining some preprocessing and
        the provided ``estimator``.
    """  # noqa: E501
    warnings.warn(
        (
            "tabular_learner will be deprecated in the next release. "
            "Equivalent functionality is available in skrub.tabular_pipeline."
        ),
        category=FutureWarning,
    )
    return tabular_pipeline(estimator, n_jobs=n_jobs)


def tabular_pipeline(estimator, *, n_jobs=None):
    """Get a simple machine-learning pipeline for tabular data.

    Given either a scikit-learn estimator or one of the special-cased strings
    ``'regressor'``, ``'regression'``, ``'classifier'``, ``'classification'``, this
    function creates a scikit-learn pipeline that extracts numeric features, imputes
    missing values and scales the data if necessary, then applies the estimator.

    .. note::
       The heuristics used by the ``tabular_pipeline``
       to define an appropriate preprocessing based on the ``estimator`` may change
       in future releases.

    .. versionchanged:: 0.6.0
        The high cardinality encoder has been changed from
        :class:`~skrub.MinHashEncoder` to :class:`~skrub.StringEncoder`.

    Parameters
    ----------
    estimator : {"regressor", "regression", "classifier", "classification"} or sklearn.base.BaseEstimator
        The estimator to use as the final step in the pipeline. Based on the type of
        estimator, the previous preprocessing steps and their respective parameters are
        chosen. The possible values are:

        - ``'regressor'`` or ``'regression'``: a
          :obj:`~sklearn.ensemble.HistGradientBoostingRegressor` is used as the final
          step;
        - ``'classifier'`` or ``'classification'``: a
          :obj:`~sklearn.ensemble.HistGradientBoostingClassifier` is used as the final
          step;
        - a scikit-learn estimator: the provided estimator is used as the final step.

    n_jobs : int, default=None
        Number of jobs to run in parallel in the :obj:`TableVectorizer` step. ``None``
        means 1 unless in a joblib ``parallel_backend`` context. ``-1`` means using all
        processors.

    Returns
    -------
    Pipeline
        A scikit-learn :obj:`~sklearn.pipeline.Pipeline` chaining some preprocessing and
        the provided ``estimator``.

    Notes
    -----

    ``tabular_pipeline`` returns a scikit-learn :obj:`~sklearn.pipeline.Pipeline` with
    several steps:

    - A :obj:`TableVectorizer` transforms the tabular data into numeric features. Its
      parameters are chosen depending on the provided ``estimator``.
    - An optional :obj:`~sklearn.impute.SimpleImputer` imputes missing values by their
      mean and adds binary columns that indicate which values were missing. This step is
      only added if the ``estimator`` cannot handle missing values itself.
    - An optional :obj:`~sklearn.preprocessing.StandardScaler` centers and rescales the
      data. This step is not added (because it is unnecessary) when the ``estimator`` is
      a tree ensemble such as random forest or gradient boosting.
    - The last step is the provided ``estimator``.

    The parameter values for the :obj:`TableVectorizer` might differ depending on the
    version of scikit-learn:

    - support for categorical features in
      :class:`~sklearn.ensemble.HistGradientBoostingClassifier` and
      :class:`~sklearn.ensemble.HistGradientBoostingRegressor` was added in scikit-learn
      1.4. Therefore, before this version, a
      :class:`~sklearn.preprocessing.OrdinalEncoder` is used for low-cardinality
      features.
    - support for missing values in :class:`~sklearn.ensemble.RandomForestClassifier`
      and :class:`~sklearn.ensemble.RandomForestRegressor` was added in scikit-learn
      1.4. Therefore, before this version, a :class:`~sklearn.impute.SimpleImputer` is
      used to impute missing values.

    Read more in the :ref:`User Guide <user_guide_tabular_pipeline>`.

    Examples
    --------
    >>> from skrub import tabular_pipeline

    We can easily get a default pipeline for regression or classification:

    >>> tabular_pipeline('regression')                                    # doctest: +SKIP
    Pipeline(steps=[('tablevectorizer',
                     TableVectorizer(high_cardinality=StringEncoder(),
                                     low_cardinality=ToCategorical())),
                    ('histgradientboostingregressor',
                     HistGradientBoostingRegressor(categorical_features='from_dtype'))])

    When requesting a ``'regression'``, the last step of the pipeline is set to a
    :obj:`~sklearn.ensemble.HistGradientBoostingRegressor`.

    >>> tabular_pipeline('classification')                                   # doctest: +SKIP
    Pipeline(steps=[('tablevectorizer',
                     TableVectorizer(high_cardinality=StringEncoder(),
                                     low_cardinality=ToCategorical())),
                    ('histgradientboostingclassifier',
                     HistGradientBoostingClassifier(categorical_features='from_dtype'))])

    When requesting a ``'classification'``, the last step of the pipeline is set to a
    :obj:`~sklearn.ensemble.HistGradientBoostingClassifier`.

    This pipeline can be applied to rich tabular data:

    >>> import pandas as pd
    >>> X = pd.DataFrame(
    ...     {
    ...         "last_visit": ["2020-01-02", "2021-04-01", "2024-12-05", "2023-08-10"],
    ...         "medication": [None, "metformin", "paracetamol", "gliclazide"],
    ...         "insulin_prescriptions": ["N/A", 13, 0, 17],
    ...         "fasting_glucose": [35, 140, 44, 137],
    ...     }
    ... )
    >>> y = [0, 1, 0, 1]
    >>> X
       last_visit   medication insulin_prescriptions  fasting_glucose
    0  2020-01-02         None                   N/A               35
    1  2021-04-01    metformin                    13              140
    2  2024-12-05  paracetamol                     0               44
    3  2023-08-10   gliclazide                    17              137

    >>> model = tabular_pipeline('classifier').fit(X, y)
    >>> model.predict(X)
    array([0, 0, 0, 0])

    Rather than using the default estimator, we can provide our own scikit-learn
    estimator:

    >>> from sklearn.linear_model import LogisticRegression
    >>> model = tabular_pipeline(LogisticRegression())
    >>> model.fit(X, y)
    Pipeline(steps=[('tablevectorizer',
                    TableVectorizer(datetime=DatetimeEncoder(periodic_encoding='spline'))),
                    ('simpleimputer', SimpleImputer(add_indicator=True)),
                    ('standardscaler', StandardScaler()),
                    ('logisticregression', LogisticRegression())])

    By applying only the first pipeline step we can see the transformed data that is
    sent to the supervised estimator (see the :obj:`TableVectorizer` documentation for
    details):

    >>> model.named_steps['tablevectorizer'].transform(X)               # doctest: +SKIP
       last_visit_year  last_visit_month  ...  insulin_prescriptions  fasting_glucose
    0           2020.0               1.0  ...                    NaN             35.0
    1           2021.0               4.0  ...                   13.0            140.0
    2           2024.0              12.0  ...                    0.0             44.0
    3           2023.0               8.0  ...                   17.0            137.0

    The parameters of the :obj:`TableVectorizer` depend on the provided ``estimator``.

    >>> tabular_pipeline(LogisticRegression())
    Pipeline(steps=[('tablevectorizer',
                    TableVectorizer(datetime=DatetimeEncoder(periodic_encoding='spline'))),
                    ('simpleimputer', SimpleImputer(add_indicator=True)),
                    ('standardscaler', StandardScaler()),
                    ('logisticregression', LogisticRegression())])

    For a :obj:`~sklearn.linear_model.LogisticRegression`, we get:

    - a default configuration of the :obj:`TableVectorizer` which is intended to work
      well for a wide variety of downstream estimators. The configuration adds
      ``spline`` periodic features to datetime columns.

    - A :obj:`~sklearn.impute.SimpleImputer`, as the
      :obj:`~sklearn.linear_model.LogisticRegression` cannot handle missing values.

    - A :obj:`~sklearn.preprocessing.StandardScaler` for centering and standard scaling
      numerical features.

    On the other hand, For the :obj:`~sklearn.ensemble.HistGradientBoostingClassifier`
    (generated with the string ``"classifier"``):

    >>> tabular_pipeline('classifier')                                   # doctest: +SKIP
    Pipeline(steps=[('tablevectorizer',
                     TableVectorizer(high_cardinality=StringEncoder(),
                                     low_cardinality=ToCategorical())),
                    ('histgradientboostingclassifier',
                     HistGradientBoostingClassifier(categorical_features='from_dtype'))])

    - A :obj:`StringEncoder` is used as the ``high_cardinality`` encoder. This encoder
      strikes a good balance between quality and performance in most situations.

    - The ``low_cardinality`` does not one-hot encode features. The
      :obj:`~sklearn.ensemble.HistGradientBoostingClassifier` has built-in support for
      categorical data which is more efficient than one-hot encoding. Therefore the
      selected encoder, :obj:`ToCategorical`, simply makes sure that those features have
      a categorical dtype so that the
      :obj:`~sklearn.ensemble.HistGradientBoostingClassifier` recognizes them as such.

    - There is no spline encoding of datetimes.

    - There is no missing-value imputation because the classifier has its own (better)
      mechanism for dealing with missing values, and no standard scaling because it is
      unnecessary for tree ensembles.
    """  # noqa: E501
    vectorizer = TableVectorizer(n_jobs=n_jobs)
    if parse_version(sklearn.__version__) < parse_version("1.4"):
        cat_feat_kwargs = {}
    else:
        cat_feat_kwargs = {"categorical_features": "from_dtype"}

    if isinstance(estimator, str):
        if estimator in ("classifier", "classification"):
            return tabular_pipeline(
                ensemble.HistGradientBoostingClassifier(**cat_feat_kwargs),
                n_jobs=n_jobs,
            )
        if estimator in ("regressor", "regression"):
            return tabular_pipeline(
                ensemble.HistGradientBoostingRegressor(**cat_feat_kwargs),
                n_jobs=n_jobs,
            )
        raise ValueError(
            "If ``estimator`` is a string it should be 'regressor', 'regression',"
            " 'classifier' or 'classification'."
        )
    if isinstance(estimator, type) and issubclass(estimator, BaseEstimator):
        raise TypeError(
            "tabular_pipeline expects a scikit-learn estimator as its first"
            f" argument. Pass an instance of {estimator.__name__} rather than the class"
            " itself."
        )
    if not isinstance(estimator, BaseEstimator):
        raise TypeError(
            "tabular_pipeline expects a scikit-learn estimator, 'regressor',"
            " or 'classifier' as its first argument."
        )

    if (
        isinstance(estimator, _HGBT_CLASSES)
        and getattr(estimator, "categorical_features", None) == "from_dtype"
    ):
        vectorizer.set_params(
            low_cardinality=ToCategorical(),
            high_cardinality=StringEncoder(),
        )
    elif isinstance(estimator, _TREE_ENSEMBLE_CLASSES):
        vectorizer.set_params(
            low_cardinality=OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            high_cardinality=StringEncoder(),
        )
    else:
        vectorizer.set_params(datetime=DatetimeEncoder(periodic_encoding="spline"))
    steps = [vectorizer]
    if not get_tags(estimator).input_tags.allow_nan:
        steps.append(SimpleImputer(add_indicator=True))
    if not isinstance(estimator, _TREE_ENSEMBLE_CLASSES):
        steps.append(StandardScaler())
    steps.append(estimator)
    return make_pipeline(*steps)
