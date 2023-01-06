# Inspired from https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tests/test_base.py # noqa
# Author: Gael Varoquaux
# License: BSD 3 clause

import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from dirty_cat import MinHashEncoder
from dirty_cat._base import BaseEstimator


#############################################################################
# A few test classes
class MyEstimator(BaseEstimator):
    def __init__(self, l1=0, empty=None):
        self.l1 = l1
        self.empty = empty


class K(BaseEstimator):
    def __init__(self, c=None, d=None):
        self.c = c
        self.d = d


class T(BaseEstimator):
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b


class NoEstimator:
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self


#############################################################################
# The tests


def test_get_params():
    test = T(K(), K)

    assert "a__d" in test.get_params(deep=True)
    assert "a__d" not in test.get_params(deep=False)

    test.set_params(a__d=2)
    assert test.a.d == 2

    with pytest.raises(ValueError):
        test.set_params(a__a=2)


def test_set_params():
    # test nested estimator parameter setting
    clf = Pipeline([("minhash", MinHashEncoder())])

    # non-existing parameter in minhash
    with pytest.raises(ValueError):
        clf.set_params(minhash__stupid_param=True)

    # we don't currently catch if the things in pipeline are estimators
    # bad_pipeline = Pipeline([("bad", NoEstimator())])
    # assert_raises(AttributeError, bad_pipeline.set_params,
    #               bad__stupid_param=True)


def test_set_params_passes_all_parameters():
    # Make sure all parameters are passed together to set_params
    # of nested estimator.

    class TestMinHash(MinHashEncoder):
        def set_params(self, **kwargs):
            super().set_params(**kwargs)
            # expected_kwargs is in test scope
            assert kwargs == expected_kwargs
            return self

    expected_kwargs = {"ngram_range": (2, 4), "n_components": 2}
    for est in [
        Pipeline([("estimator", TestMinHash())]),
        GridSearchCV(TestMinHash(), {}),
    ]:
        est.set_params(estimator__ngram_range=(2, 4), estimator__n_components=2)


def test_set_params_updates_valid_params():
    # Check that set_params tries to set SVC().C, not
    # DecisionTreeClassifier().C
    gscv = GridSearchCV(DecisionTreeClassifier(), {})
    gscv.set_params(estimator=SVC(), estimator__C=42.0)
    assert gscv.estimator.C == 42.0


def test_raises_on_get_params_non_attribute():
    class MyEstimator(BaseEstimator):
        def __init__(self, param=5):
            pass

        def fit(self, X, y=None):
            return self

    est = MyEstimator()
    msg = "'MyEstimator' object has no attribute 'param'"

    with pytest.raises(AttributeError, match=msg):
        est.get_params()
