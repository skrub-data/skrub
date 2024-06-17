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


def table_learner(estimator, n_jobs=None):
    """Get a simple machine-learning pipeline that should work well in many cases.

    This function returns a scikit-learn :obj:`~sklearn.pipeline.Pipeline` that
    combines a :obj:`TableVectorizer`, a :obj:`~sklearn.impute.SimpleImputer`
    if missing values are not handled by the provided ``estimator``, and
    finally the ``estimator`` itself.

    This pipeline is simple but (depending on the chosen ``estimator``) should
    provide a strong baseline for many learning problems. It can handle tabular
    input and complex data such as categories, text or datetimes.

    Parameters
    ----------
    estimator : str or scikit-learn estimator
        The estimator to use as the final step in the pipeline. Appropriate
        choices are made for previous step depending on the ``estimator``. Can
        be the string ``"regressor"`` to use a
        :obj:`~sklearn.ensemble.HistGradientBoostingRegressor` or
        ``"classifier"`` to use a
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
    >>> from skrub import table_learner

    We can easily get a default pipeline for classification or regression:

    >>> table_learner('regressor')                # doctest: +SKIP
    Pipeline(steps=[('tablevectorizer',
                     TableVectorizer(high_cardinality_transformer=MinHashEncoder(),
                                     low_cardinality_transformer=ToCategorical())),
                    ('histgradientboostingregressor',
                     HistGradientBoostingRegressor(categorical_features='from_dtype'))])

    This pipeline can handle complex, tabular data:

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
    >>> model = table_learner('classifier').fit(X, y)
    >>> model.predict(X)
    array([False, False, False, False])

    By applying only the first pipeline step we can see the transformed data
    that is sent to the supervised estimator (see the :obj:`TableVectorizer`
    documentation for details):

    >>> model.named_steps['tablevectorizer'].transform(X)   # doctest: +SKIP
       last_visit_year  last_visit_month  ...  insulin_prescriptions  fasting_glucose
    0           2020.0               1.0  ...                    NaN             35.0
    1           2021.0               4.0  ...                   13.0            140.0
    2           2024.0              12.0  ...                    0.0             44.0
    3           2023.0               8.0  ...                   17.0            137.0

    The default pipeline combines a :obj:`TableVectorizer` and a
    :obj:`~sklearn.ensemble.HistGradientBoostingRegressor` (or
    :obj:`~sklearn.ensemble.HistGradientBoostingClassifier` for
    classification).

    The parameters of the :obj:`TableVectorizer` differ from the default ones:

      - A :obj:`MinHashEncoder` is used as the
        ``high_cardinality_transformer``. This encoder provides good
        performance when the supervised estimator is based on a decision tree
        or ensemble of trees, as is the case for the
        :obj:`~sklearn.ensemble.HistGradientBoostingClassifier`. Unlike the
        default :obj:`GapEncoder`, the :obj:`MinHashEncoder` does not produce
        interpretable features. However, it is much faster and uses less
        memory.

      - The ``low_cardinality_transformer`` does not one-hot encode features.
        The :obj:`~sklearn.ensemble.HistGradientBoostingRegressor` has built-in
        support for categorical data which is more efficient than one-hot
        encoding. Therefore the selected encoder, :obj:`ToCategorical`, simply
        makes sure that those features have a categorical dtype so that the
        :obj:`~sklearn.ensemble.HistGradientBoostingRegressor` recognizes them
        as such.

    We can also choose which predictor (the final step in the pipeline) to use,
    and appropriate choices will be made for the :obj:`TableVectorizer`. Moreover,
    unless the predictor has built-in support for null values, an imputation
    step will be added.

    >>> from sklearn.linear_model import Ridge
    >>> model = table_learner(Ridge(solver='lsqr'))
    >>> model.fit(X, y)
    Pipeline(steps=[('tablevectorizer', TableVectorizer()),
                    ('simpleimputer', SimpleImputer()),
                    ('ridge', Ridge(solver='lsqr'))])

    Here, different choices were made for the :obj:`TableVectorizer` -- in
    particular the categorical column ``"c"`` is one-hot encoded, because the
    :obj:`~sklearn.linear_model.Ridge` regressor lacks the
    :obj:`~sklearn.ensemble.HistGradientBoostingRegressor`'s built-in support
    for categorical variables.

    >>> model.named_steps['tablevectorizer'].transform(X)
       last_visit_year  last_visit_month  ...  insulin_prescriptions  fasting_glucose
    0           2020.0               1.0  ...                    NaN             35.0
    1           2021.0               4.0  ...                   13.0            140.0
    2           2024.0              12.0  ...                    0.0             44.0
    3           2023.0               8.0  ...                   17.0            137.0

    Moreover, as :obj:`~sklearn.linear_model.Ridge` does not handle missing
    values, a step was added to perform mean imputation. Therefore the data
    seen by the final predictor actually looks like this (note: scikit-learn
    :obj:`~sklearn.pipeline.Pipeline` can be sliced to produce another
    ``Pipeline`` containing only the specified steps):

    >>> model[:2].transform(X)
    array([[2.0200000e+03, 1.0000000e+00, 2.0000000e+00, 1.5779232e+09,
            0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00,
            1.0000000e+01, 3.5000000e+01],
           [2.0210000e+03, 4.0000000e+00, 1.0000000e+00, 1.6172352e+09,
            0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 0.0000000e+00,
            1.3000000e+01, 1.4000000e+02],
           [2.0240000e+03, 1.2000000e+01, 5.0000000e+00, 1.7333568e+09,
            0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00,
            0.0000000e+00, 4.4000000e+01],
           [2.0230000e+03, 8.0000000e+00, 1.0000000e+01, 1.6916256e+09,
            1.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
            1.7000000e+01, 1.3700000e+02]], dtype=float32)
    """  # noqa: E501
    vectorizer = TableVectorizer(n_jobs=n_jobs)
    if parse_version(sklearn.__version__) < parse_version("1.4"):
        cat_feat_kwargs = {}
    else:
        cat_feat_kwargs = {"categorical_features": "from_dtype"}

    if isinstance(estimator, str):
        if estimator == "classifier":
            return table_learner(
                ensemble.HistGradientBoostingClassifier(**cat_feat_kwargs),
                n_jobs=n_jobs,
            )
        if estimator == "regressor":
            return table_learner(
                ensemble.HistGradientBoostingRegressor(**cat_feat_kwargs),
                n_jobs=n_jobs,
            )
        raise ValueError(
            "If ``predictor`` is a string it should be 'regressor' or 'classifier'."
        )
    if isinstance(estimator, type) and issubclass(estimator, BaseEstimator):
        raise TypeError(
            "table_learner expects a scikit-learn estimator as its first"
            f" argument. Pass an instance of {estimator.__name__} rather than the class"
            " itself."
        )
    if not isinstance(estimator, BaseEstimator):
        raise TypeError(
            "table_learner expects a scikit-learn estimator, 'regressor',"
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
    if estimator._get_tags().get("allow_nan", False):
        steps = (vectorizer, estimator)
    else:
        steps = (vectorizer, SimpleImputer(), estimator)
    return make_pipeline(*steps)
