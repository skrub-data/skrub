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


def make_tabular_pipeline(predictor, n_jobs=None):
    """Get a simple machine-learning pipeline that should work well in many cases.

    This function returns a scikit-learn :obj:`~sklearn.pipeline.Pipeline` that
    combines a :obj:`TableVectorizer`, a :obj:`~sklearn.impute.SimpleImputer`
    if missing values are not handled by the provided ``predictor``, and
    finally the ``predictor`` itself.

    This pipeline is simple but (depending on the chosen ``predictor``) should
    provide a strong baseline for many learning problems. It can handle tabular
    input and complex data such as categories, text or datetimes.

    Parameters
    ----------
    predictor : str or scikit-learn estimator
        The estimator to use as the final step in the pipeline. Appropriate
        choices are made for previous step depending on the ``predictor``. Can
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
        preprocessing and the provided ``predictor``.

    Examples
    --------
    >>> from skrub import make_tabular_pipeline

    We can easily get a default pipeline for classification or regression:

    >>> make_tabular_pipeline('regressor')                # doctest: +SKIP
    Pipeline(steps=[('tablevectorizer',
                     TableVectorizer(high_cardinality_transformer=MinHashEncoder(),
                                     low_cardinality_transformer=ToCategorical())),
                    ('histgradientboostingregressor',
                     HistGradientBoostingRegressor(categorical_features='from_dtype'))])

    This pipeline can handle complex, tabular data:

    >>> import pandas as pd
    >>> X = pd.DataFrame({'a': ['2020-01-02', '2021-04-01'], 'b': [None, 11], 'c': ['a', 'b']})
    >>> y = [True, False]
    >>> X
                a     b  c
    0  2020-01-02   NaN  a
    1  2021-04-01  11.0  b
    >>> model = make_tabular_pipeline('regressor').fit(X, y)
    >>> model.predict(X)
    array([0.5, 0.5])

    By applying only the first pipeline step we can see the transformed data
    that is sent to the supervised predictor (see the :obj:`TableVectorizer`
    documentation for details):

    >>> model.named_steps['tablevectorizer'].transform(X)   # doctest: +SKIP
       a_year  a_month  a_day  a_total_seconds     b  c
    0  2020.0      1.0    2.0     1.577923e+09   NaN  a
    1  2021.0      4.0    1.0     1.617235e+09  11.0  b

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
    >>> model = make_tabular_pipeline(Ridge(solver='lsqr'))
    >>> model.fit(X, y)
    Pipeline(steps=[('tablevectorizer', TableVectorizer()),
                    ('simpleimputer', SimpleImputer()),
                    ('ridge', Ridge(solver='lsqr'))])

    Here, different choices were made for the :obj:`TableVectorizer` -- in
    particular the categorical column ``"c"`` is one-hot encoded, because the
    :obj:`~sklearn.linear_model.Ridge` regressor lacks the
    :obj:`~sklearn.ensemble.HistGradientBoostingRegressor`'s bult-in support
    for categorical variables.

    >>> model.named_steps['tablevectorizer'].transform(X)
       a_year  a_month  a_day  a_total_seconds     b  c_b
    0  2020.0      1.0    2.0     1.577923e+09   NaN  0.0
    1  2021.0      4.0    1.0     1.617235e+09  11.0  1.0

    Moreover, as :obj:`~sklearn.linear_model.Ridge` does not handle missing
    values, a step was added to perform mean imputation. Therefore the data
    seen by the final predictor actually looks like this (note scikit-learn
    :obj:`~sklearn.pipeline.Pipeline` can be sliced to produce another
    ``Pipeline`` containing only the specified steps):

    >>> model[:2].transform(X)
    array([[2.0200000e+03, 1.0000000e+00, 2.0000000e+00, 1.5779232e+09,
            1.1000000e+01, 0.0000000e+00],
           [2.0210000e+03, 4.0000000e+00, 1.0000000e+00, 1.6172352e+09,
            1.1000000e+01, 1.0000000e+00]], dtype=float32)

    # noqa
    """
    vectorizer = TableVectorizer(n_jobs=n_jobs)
    if parse_version(sklearn.__version__) < parse_version("1.4"):
        cat_feat_kwargs = {}
    else:
        cat_feat_kwargs = {"categorical_features": "from_dtype"}
    match predictor:
        case "classifier":
            return make_tabular_pipeline(
                ensemble.HistGradientBoostingClassifier(**cat_feat_kwargs),
                n_jobs=n_jobs,
            )
        case "regressor":
            return make_tabular_pipeline(
                ensemble.HistGradientBoostingRegressor(**cat_feat_kwargs),
                n_jobs=n_jobs,
            )
        case str():
            raise ValueError(
                "If ``predictor`` is a string it should be 'regressor' or 'classifier'"
            )
        case ensemble.HistGradientBoostingClassifier(
            categorical_features="from_dtype"
        ) | ensemble.HistGradientBoostingRegressor(categorical_features="from_dtype"):
            vectorizer.set_params(
                low_cardinality_transformer=ToCategorical(),
                high_cardinality_transformer=MinHashEncoder(),
            )
        case (
            ensemble.HistGradientBoostingClassifier()
            | ensemble.HistGradientBoostingRegressor()
            | ensemble.RandomForestClassifier()
            | ensemble.RandomForestRegressor()
        ):
            vectorizer.set_params(
                low_cardinality_transformer=OrdinalEncoder(),
                high_cardinality_transformer=MinHashEncoder(),
            )
        case BaseEstimator():
            pass
        case type() as cls if issubclass(cls, BaseEstimator):
            raise TypeError(
                "make_tabular_pipeline expects a scikit-learn estimator as its first"
                f" argument. Pass an instance of {cls.__name__} rather than the class"
                " itself."
            )
        case _:
            raise TypeError(
                "make_tabular_pipeline expects a scikit-learn estimator, 'regressor',"
                " or 'classifier' as its first argument."
            )

    if predictor._get_tags().get("allow_nan", False):
        steps = (vectorizer, predictor)
    else:
        steps = (vectorizer, SimpleImputer(), predictor)
    return make_pipeline(*steps)
