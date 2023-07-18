from sklearn.utils.estimator_checks import parametrize_with_checks

from skrub import (
    DatetimeEncoder,
    FeatureAugmenter,
    GapEncoder,
    MinHashEncoder,
    SimilarityEncoder,
    TableVectorizer,
    TargetEncoder,
)


def _tested_estimators():
    for Estimator in [
        # DatetimeEncoder,
        # FeatureAugmenter,
        # GapEncoder,
        # MinHashEncoder,
        # SimilarityEncoder,
        # TableVectorizer,
        TargetEncoder,
    ]:
        yield Estimator()


@parametrize_with_checks(list(_tested_estimators()))
def test_estimators_compatibility_sklearn(estimator, check, request):
    check(estimator)
