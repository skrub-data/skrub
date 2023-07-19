import numpy as np
from scipy import sparse

import sklearn
from sklearn.metrics.pairwise import linear_kernel, pairwise_distances
from sklearn.utils.estimator_checks import (
    _is_pairwise_metric,
    parametrize_with_checks,
)
from sklearn.utils._tags import _safe_tags

from skrub import (
    DatetimeEncoder,
    FeatureAugmenter,
    GapEncoder,
    MinHashEncoder,
    SimilarityEncoder,
    TableVectorizer,
    TargetEncoder,
)


def _enforce_estimator_tags_X_monkey_patch(estimator, X, kernel=linear_kernel):
    # Estimators with `1darray` in `X_types` tag only accept
    # X of shape (`n_samples`,)
    if "1darray" in _safe_tags(estimator, key="X_types"):
        X = X[:, 0]
    # Estimators with a `requires_positive_X` tag only accept
    # strictly positive data
    if _safe_tags(estimator, key="requires_positive_X"):
        X = X - X.min()
    if "categorical" in _safe_tags(estimator, key="X_types"):
        X = np.round((X - X.min()))
        if "string" in _safe_tags(estimator, key="X_types"):
            X = X.astype(object)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    X[i, j] = str(X[i, j])
        elif _safe_tags(estimator, key="allow_nan"):
            X = X.astype(np.float64)
        else:
            X = X.astype(np.int32)

    if estimator.__class__.__name__ == "SkewedChi2Sampler":
        # SkewedChi2Sampler requires X > -skewdness in transform
        X = X - X.min()

    # Pairwise estimators only accept
    # X of shape (`n_samples`, `n_samples`)
    if _is_pairwise_metric(estimator):
        X = pairwise_distances(X, metric="euclidean")
    elif _safe_tags(estimator, key="pairwise"):
        X = kernel(X, X)
    return X


sklearn.utils.estimator_checks._enforce_estimator_tags_X = (
    _enforce_estimator_tags_X_monkey_patch
)


def _tested_estimators():
    for Estimator in [
        # DatetimeEncoder,
        # FeatureAugmenter,
        GapEncoder,
        # MinHashEncoder,
        # SimilarityEncoder,
        # TableVectorizer,
        # TargetEncoder,
    ]:
        yield Estimator()


@parametrize_with_checks(list(_tested_estimators()))
def test_estimators_compatibility_sklearn(estimator, check, request):
    check(estimator)
