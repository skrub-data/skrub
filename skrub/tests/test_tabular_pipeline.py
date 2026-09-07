import pytest
from sklearn import ensemble
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from skrub import (
    SquashingScaler,
    StringEncoder,
    TableVectorizer,
    ToCategorical,
    tabular_pipeline,
)


@pytest.mark.parametrize(
    "learner_kind", ["regressor", "regression", "classifier", "classification"]
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


class TabICLClassifier(BaseEstimator):
    """Dummy class which pretends to be `tabicl.TabICLClassifier`"""

    pass


class TabICLRegressor(BaseEstimator):
    """Dummy class which pretends to be `tabicl.TabICLRegressor`"""

    pass


@pytest.fixture(
    scope="module",
    params=[pytest.param(TabICLClassifier()), pytest.param(TabICLRegressor())],
    ids=["TabICLClassifier-instance", "TabICLRegressor-instance"],
)
def tabicl_estimator(request):
    return request.param


def test_tabicl_pipeline(tabicl_estimator):
    p = tabular_pipeline(tabicl_estimator)
    tv, learner = (e for _, e in p.steps)
    assert isinstance(tv, TableVectorizer)

    assert tv.low_cardinality == "passthrough"
    assert isinstance(tv.high_cardinality, StringEncoder)
    assert tv.cardinality_threshold == 10
    assert tv.datetime.periodic_encoding == "spline"
