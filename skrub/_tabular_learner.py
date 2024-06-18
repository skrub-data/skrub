import sklearn
from sklearn import ensemble
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.fixes import parse_version

from ._minhash_encoder import MinHashEncoder
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


def tabular_learner(estimator, n_jobs=None):
    """Get a simple machine-learning pipeline for tabular data.

    Given a scikit-learn ``estimator``, this function creates a
    machine-learning pipeline that preprocesses tabular data to extract numeric
    features and impute missing values if necessary, then applies the
    ``estimator``.

    Instead of an actual estimator, ``estimator`` can also be the string
    ``'regressor'`` or ``'classifier'`` to use a
    :obj:`~sklearn.ensemble.HistGradientBoostingRegressor` or a
    :obj:`~sklearn.ensemble.HistGradientBoostingClassifier` with default
    parameters.

    ``tabular_learner`` returns a scikit-learn :obj:`~sklearn.pipeline.Pipeline`
    with several steps:

    - a :obj:`TableVectorizer` transforms the tabular data into numeric
      features. Its parameters are chosen depending on the provided
      ``estimator``.
    - an optional :obj:`~sklearn.impute.SimpleImputer` imputes missing values
      by their mean and adds binary columns that indicate which values were
      missing. This step is only added if the ``estimator`` cannot handle
      missing values itself.
    - the last step is the provided ``estimator``.

    **Note:** ``tabular_learner`` is a recent addition and the heuristics used
    to define an appropriate preprocessing based on the ``estimator`` may change
    in future releases.

    Parameters
    ----------
    estimator : str or scikit-learn estimator
        The estimator to use as the final step in the pipeline. Appropriate
        choices are made for previous steps depending on the ``estimator``. Can
        be the string ``'regressor'`` to use a
        :obj:`~sklearn.ensemble.HistGradientBoostingRegressor` or
        ``'classifier'`` to use a
        :obj:`~sklearn.ensemble.HistGradientBoostingClassifier`.

    n_jobs : int, default=None
        Number of jobs to run in parallel in the :obj:`TableVectorizer` step.
        ``None`` means 1 unless in a joblib ``parallel_backend`` context.
        ``-1`` means using all processors.

    Returns
    -------
    Pipeline
        A scikit-learn :obj:`~sklearn.pipeline.Pipeline` chaining some
        preprocessing and the provided ``estimator``.

    Examples
    --------
    >>> from skrub import tabular_learner

    We can easily get a default pipeline for classification or regression:

    >>> tabular_learner('regressor')                             # doctest: +SKIP
    Pipeline(steps=[('tablevectorizer',
                     TableVectorizer(high_cardinality_transformer=MinHashEncoder(),
                                     low_cardinality_transformer=ToCategorical())),
                    ('histgradientboostingregressor',
                     HistGradientBoostingRegressor(categorical_features='from_dtype'))])

    >>> tabular_learner('classifier')                            # doctest: +SKIP
    Pipeline(steps=[('tablevectorizer',
                     TableVectorizer(high_cardinality_transformer=MinHashEncoder(),
                                     low_cardinality_transformer=ToCategorical())),
                    ('histgradientboostingclassifier',
                     HistGradientBoostingClassifier(categorical_features='from_dtype'))])

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
    >>> y = [False, True, False, True]
    >>> X
       last_visit   medication insulin_prescriptions  fasting_glucose
    0  2020-01-02         None                   N/A               35
    1  2021-04-01    metformin                    13              140
    2  2024-12-05  paracetamol                     0               44
    3  2023-08-10   gliclazide                    17              137

    >>> model = tabular_learner('classifier').fit(X, y)
    >>> model.predict(X)
    array([False, False, False, False])

    If we pass ``estimator="regressor"``, the last step of the pipeline is a
    :obj:`~sklearn.ensemble.HistGradientBoostingRegressor`. If we pass
    ``estimator="classifier"``, it is a
    :obj:`~sklearn.ensemble.HistGradientBoostingClassifier`. Rather than using
    the default estimator, we can provide our own:

    >>> from sklearn.linear_model import LogisticRegression
    >>> model = tabular_learner(LogisticRegression())
    >>> model.fit(X, y)
    Pipeline(steps=[('tablevectorizer', TableVectorizer()),
                    ('simpleimputer', SimpleImputer(add_indicator=True)),
                    ('logisticregression', LogisticRegression())])

    By applying only the first pipeline step we can see the transformed data
    that is sent to the supervised estimator (see the :obj:`TableVectorizer`
    documentation for details):

    >>> model.named_steps['tablevectorizer'].transform(X)            # doctest: +SKIP
       last_visit_year  last_visit_month  ...  insulin_prescriptions  fasting_glucose
    0           2020.0               1.0  ...                    NaN             35.0
    1           2021.0               4.0  ...                   13.0            140.0
    2           2024.0              12.0  ...                    0.0             44.0
    3           2023.0               8.0  ...                   17.0            137.0

    The parameters of the :obj:`TableVectorizer` depend on the provided
    ``estimator``.

    >>> tabular_learner(LogisticRegression())
    Pipeline(steps=[('tablevectorizer', TableVectorizer()),
                    ('simpleimputer', SimpleImputer(add_indicator=True)),
                    ('logisticregression', LogisticRegression())])

    We see that for the :obj:`~sklearn.linear_model.LogisticRegression` we get
    the default configuration of the ``TableVectorizer`` which is intended to
    work well for a wide variety of downstream estimators. Moreover, as the
    ``LogisticRegression`` cannot handle missing values, an imputation step is
    added.

    On the other hand, For the :obj:`~sklearn.ensemble.HistGradientBoostingClassifier`:

    >>> tabular_learner('classifier')                             # doctest: +SKIP
    Pipeline(steps=[('tablevectorizer',
                     TableVectorizer(high_cardinality_transformer=MinHashEncoder(),
                                     low_cardinality_transformer=ToCategorical())),
                    ('histgradientboostingclassifier',
                     HistGradientBoostingClassifier(categorical_features='from_dtype'))])

    - A :obj:`MinHashEncoder` is used as the
      ``high_cardinality_transformer``. This encoder provides good
      performance when the supervised estimator is based on a decision tree
      or ensemble of trees, as is the case for the
      :obj:`~sklearn.ensemble.HistGradientBoostingClassifier`. Unlike the
      default :obj:`GapEncoder`, the :obj:`MinHashEncoder` does not produce
      interpretable features. However, it is much faster and uses less
      memory.

    - The ``low_cardinality_transformer`` does not one-hot encode features.
      The :obj:`~sklearn.ensemble.HistGradientBoostingClassifier` has built-in
      support for categorical data which is more efficient than one-hot
      encoding. Therefore the selected encoder, :obj:`ToCategorical`, simply
      makes sure that those features have a categorical dtype so that the
      :obj:`~sklearn.ensemble.HistGradientBoostingClassifier` recognizes them
      as such.

    - There is no missing-value imputation because the classifier has its own
      (better) mechanism for dealing with missing values.
    """  # noqa: E501
    vectorizer = TableVectorizer(n_jobs=n_jobs)
    if parse_version(sklearn.__version__) < parse_version("1.4"):
        cat_feat_kwargs = {}
    else:
        cat_feat_kwargs = {"categorical_features": "from_dtype"}

    if isinstance(estimator, str):
        if estimator == "classifier":
            return tabular_learner(
                ensemble.HistGradientBoostingClassifier(**cat_feat_kwargs),
                n_jobs=n_jobs,
            )
        if estimator == "regressor":
            return tabular_learner(
                ensemble.HistGradientBoostingRegressor(**cat_feat_kwargs),
                n_jobs=n_jobs,
            )
        raise ValueError(
            "If ``estimator`` is a string it should be 'regressor' or 'classifier'."
        )
    if isinstance(estimator, type) and issubclass(estimator, BaseEstimator):
        raise TypeError(
            "tabular_learner expects a scikit-learn estimator as its first"
            f" argument. Pass an instance of {estimator.__name__} rather than the class"
            " itself."
        )
    if not isinstance(estimator, BaseEstimator):
        raise TypeError(
            "tabular_learner expects a scikit-learn estimator, 'regressor',"
            " or 'classifier' as its first argument."
        )

    if (
        isinstance(estimator, _HGBT_CLASSES)
        and getattr(estimator, "categorical_features", None) == "from_dtype"
    ):
        vectorizer.set_params(
            low_cardinality_transformer=ToCategorical(),
            high_cardinality_transformer=MinHashEncoder(),
        )
    elif isinstance(estimator, _TREE_ENSEMBLE_CLASSES):
        vectorizer.set_params(
            low_cardinality_transformer=OrdinalEncoder(),
            high_cardinality_transformer=MinHashEncoder(),
        )
    if hasattr(estimator, "_get_tags") and estimator._get_tags().get(
        "allow_nan", False
    ):
        steps = (vectorizer, estimator)
    else:
        steps = (vectorizer, SimpleImputer(add_indicator=True), estimator)
    return make_pipeline(*steps)
