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
    """Get a simple-machine learning pipeline that should work well in many cases.

    Examples
    --------
    >>> from skrub import make_tabular_pipeline

    We can get a simple pipeline that should provide a strong baseline either
    for classification or regression:

    >>> make_tabular_pipeline('regressor')
    Pipeline(steps=[('tablevectorizer',
                     TableVectorizer(high_cardinality_transformer=MinHashEncoder(),
                                     low_cardinality_transformer=ToCategorical())),
                    ('histgradientboostingregressor',
                     HistGradientBoostingRegressor(categorical_features='from_dtype'))])

    We see that the default pipeline combines a ``TableVectorizer`` and a
    ``HistGradientBoostingRegressor`` (or ``HistGradientBoostingClassifier`` for
    classification).

    The parameters of the ``TableVectorizer`` differ from the default ones:

      - A ``MinHashEncoder`` is used as the ``high_cardinality_transformer``.
        This encoder provides good performance when the supervised estimator is
        based on a decision tree or ensemble of trees, as is the case for the
        ``HistGradientBoostingClassifier``. Unlike the default ``GapEncoder``,
        the ``MinHashEncoder`` does not produce interpretable features. However,
        it is much faster and uses less memory.

      - The ``low_cardinality_transformer`` does not one-hot encode features.
        The ``HistGradientBoostingRegressor`` has built-in support for
        categorical data which is more efficient than one-hot encoding.
        Therefore the selected encoder, ``ToCategorical``, simply makes sure
        that those features have a categorical dtype so that the
        ``HistGradientBoostingRegressor`` recognizes them as such.

    We can also choose which predictor (the final step in the pipeline) to use,
    and appropriate choices will be made for the ``TableVectorizer``. Moreover,
    unless the predictor has built-in support for null values, an imputation
    step will be added.

    >>> from sklearn.linear_model import Ridge
    >>> make_tabular_pipeline(Ridge())
    Pipeline(steps=[('tablevectorizer', TableVectorizer()),
                    ('simpleimputer', SimpleImputer()), ('ridge', Ridge())])
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
        case type(cls) if issubclass(cls, BaseEstimator):
            raise TypeError("pass an estimator instance not the class")
        case _:
            raise TypeError("pass a scikit-learn estimator")

    if predictor._get_tags().get("allow_nan", False):
        steps = (vectorizer, predictor)
    else:
        steps = (vectorizer, SimpleImputer(), predictor)
    return make_pipeline(*steps)
