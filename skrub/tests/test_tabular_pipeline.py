import numpy as np
import pandas as pd
import pytest
from sklearn import ensemble
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
    with pytest.raises(TypeError, match=".*Pass an instance"):
        tabular_pipeline(ensemble.HistGradientBoostingRegressor)
    with pytest.raises(
        TypeError, match=".*expects a scikit-learn compatible estimator"
    ):
        tabular_pipeline(object())


def test_missing_required_attribute():
    """Test that a TypeError is raised when the estimator does not have one of the
    attributes required of a scikit learn-compatible estimator"""

    class MissingSetParams:
        def fit(self, X, y=None):
            return self

        def predict(self, X):
            return np.zeros(X.shape[0])

        def get_params(self):
            return {}

    with pytest.raises(
        TypeError, match=".*expects a scikit-learn compatible estimator.*set_params"
    ):
        tabular_pipeline(MissingSetParams())


def test_required_attribute_is_not_callable():
    """Test that a TypeError is raised when the estimator has all of the required
    attributes, but one of them is not callable"""

    class PredictNotCallable:
        def fit(self, X, y=None):
            return self

        predict = 1

        def get_params(self):
            return {}

        def set_params(self, **params):
            return self

    with pytest.raises(
        TypeError, match=".*expects a scikit-learn compatible estimator.*predict"
    ):
        tabular_pipeline(PredictNotCallable())


class Regressor:
    """Dummy regressor used for tests"""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])

    def get_params(self):
        return {}

    def set_params(self, **params):
        return self


def test_sklearn_compatible_learner_returns_correct_pipeline():
    """Test that no error is raised when the estimate have both `get_params`
    and `set_params` attributes"""
    pipeline = tabular_pipeline(Regressor())
    X = pd.DataFrame({"feature": [1, 2, 3]})
    pipeline.fit(X)


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


def test_tree_ensemble_treatment_for_any_random_forest():
    """Test that special treatment for tree ensemble models is applied when
    substring 'RandomForest' appears in estimator class name"""

    class IAmARandomForestEstimator(Regressor):
        pass

    original_learner = IAmARandomForestEstimator()
    p = tabular_pipeline(original_learner)
    _, tv = p.steps[0]
    _, learner = p.steps[-1]
    assert learner is original_learner
    assert isinstance(tv.high_cardinality, StringEncoder)
    assert isinstance(tv.low_cardinality, OrdinalEncoder)
    assert tv.datetime.periodic_encoding is None


def test_tree_ensemble_treatment_for_xgboost():
    """Test that special treatment for tree ensemble models is applied when
    substring 'XGB' appears in estimator class name"""

    class IAmXGB(Regressor):
        pass

    original_learner = IAmXGB()
    p = tabular_pipeline(original_learner)
    _, tv = p.steps[0]
    _, learner = p.steps[-1]
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


class TabICLClassifier(Regressor):
    """Dummy class which pretends to be `tabicl.TabICLClassifier`"""

    pass


class TabICLRegressor(Regressor):
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
