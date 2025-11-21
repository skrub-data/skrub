import pytest
import sklearn
from sklearn import ensemble
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils.fixes import parse_version

from skrub import (
    SquashingScaler,
    StringEncoder,
    TableVectorizer,
    ToCategorical,
    tabular_learner,
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
    if parse_version(sklearn.__version__) < parse_version("1.4"):
        assert isinstance(tv.low_cardinality, OrdinalEncoder)
    else:
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
    if parse_version(sklearn.__version__) < parse_version("1.4"):
        tv, impute, learner = (e for _, e in p.steps)
        assert isinstance(impute, SimpleImputer)
    else:
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
    if parse_version(sklearn.__version__) < parse_version("1.4"):
        return
    p = tabular_pipeline(
        ensemble.HistGradientBoostingRegressor(categorical_features="from_dtype")
    )
    assert isinstance(p.named_steps["tablevectorizer"].low_cardinality, ToCategorical)


def test_warning_table_learner():
    msg = "tabular_learner will be deprecated in the next release"
    with pytest.raises(FutureWarning, match=msg):
        _ = tabular_learner("regressor")
