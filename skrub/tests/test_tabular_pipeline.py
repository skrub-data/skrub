import pytest
from sklearn import ensemble
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from skrub import (
    SquashingScaler,
    StringEncoder,
    TableVectorizer,
    ToCategorical,
    tabular_pipeline,
)


@pytest.mark.parametrize(
    "learner_kind",
    [
        "regressor",
        "regression",
        "classifier",
        "classification",
    ],
)
def test_default_pipeline(learner_kind):
    p = tabular_pipeline(learner_kind)
    tv, learner = (e for _, e in p.steps)
    assert isinstance(tv, TableVectorizer)
    assert isinstance(tv.high_cardinality, StringEncoder)
    assert isinstance(tv.low_cardinality, ToCategorical)
    assert learner.categorical_features == "from_dtype"
    if learner_kind in ("regressor", "regression"):
        assert isinstance(learner, ensemble.HistGradientBoostingRegressor)
    else:
        assert isinstance(learner, ensemble.HistGradientBoostingClassifier)


def test_bad_learner():
    with pytest.raises(
        ValueError,
        match=".*should be 'regressor', 'regression', 'classifier' or 'classification'",
    ):
        tabular_pipeline("bad")
    with pytest.raises(
        TypeError, match=".*Pass an instance of HistGradientBoostingRegressor"
    ):
        tabular_pipeline(ensemble.HistGradientBoostingRegressor)
    with pytest.raises(TypeError, match=".*expects a scikit-learn estimator"):
        tabular_pipeline(object())


def test_linear_learner():
    original_learner = Ridge()
    p = tabular_pipeline(original_learner)
    tv, imputer, scaler, learner = (e for _, e in p.steps)
    assert learner is original_learner
    assert isinstance(tv.high_cardinality, StringEncoder)
    assert isinstance(tv.low_cardinality, OneHotEncoder)
    assert isinstance(imputer, SimpleImputer)
    assert isinstance(scaler, SquashingScaler)
    assert tv.datetime.periodic_encoding == "spline"


def test_tree_learner():
    original_learner = ensemble.RandomForestClassifier()
    p = tabular_pipeline(original_learner)
    tv, learner = (e for _, e in p.steps)
    assert learner is original_learner
    assert isinstance(tv.high_cardinality, StringEncoder)
    assert isinstance(tv.low_cardinality, OrdinalEncoder)
    assert tv.datetime.periodic_encoding is None


def test_from_dtype():
    p = tabular_pipeline(
        ensemble.HistGradientBoostingRegressor(categorical_features=())
    )
    assert isinstance(p.named_steps["tablevectorizer"].low_cardinality, OrdinalEncoder)
    p = tabular_pipeline(
        ensemble.HistGradientBoostingRegressor(categorical_features="from_dtype")
    )
    assert isinstance(p.named_steps["tablevectorizer"].low_cardinality, ToCategorical)


def test_skpipeline_learner():
    original_learner = LogisticRegression()
    sk_pipeline = Pipeline([("pca", PCA()), ("clf", original_learner)])
    tab_pipeline = tabular_pipeline(sk_pipeline)
    assert len([e for _, e in tab_pipeline.steps]) == 5
    tv, imputer, scaler, pca, learner = (e for _, e in tab_pipeline.steps)
    assert learner is original_learner
    assert isinstance(pca, PCA)
