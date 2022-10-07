from sklearn.utils.estimator_checks import check_estimator

from dirty_cat import (
    DatetimeEncoder,
    GapEncoder,
    MinHashEncoder,
    SimilarityEncoder,
    TargetEncoder,
)


def test_sklearn_compatible_DatetimeEncoder():
    check_estimator(DatetimeEncoder())


def test_sklearn_compatible_GapEncoder():
    check_estimator(GapEncoder())


def test_sklearn_compatible_MinHashEncoder():
    check_estimator(MinHashEncoder())


def test_sklearn_compatible_SimilarityEncoder():
    check_estimator(SimilarityEncoder())


# def test_sklearn_compatible_SuperVectorizer():
#    check_estimator(SuperVectorizer())


def test_sklearn_compatible_TargetEncoder():
    check_estimator(TargetEncoder())
