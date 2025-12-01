import numpy as np
import pytest
import sklearn
from sklearn.metrics.pairwise import linear_kernel, pairwise_distances
from sklearn.utils.estimator_checks import _is_pairwise_metric, parametrize_with_checks

from skrub import (  # isort:skip
    DatetimeEncoder,
    # Joiner,
    GapEncoder,
    MinHashEncoder,
    SimilarityEncoder,
    TableVectorizer,
)
from skrub._sklearn_compat import get_tags


def _enforce_estimator_tags_X_monkey_patch(
    estimator, X, X_test=None, kernel=linear_kernel
):
    """Monkey patch scikit-learn function to create a specific case where to enforce
    having only strings with some encoders.
    """
    tags = get_tags(estimator)
    if tags.input_tags.one_d_array:
        X = X[:, 0]
        if X_test is not None:
            X_test = X_test[:, 0]  # pragma: no cover
    # Estimators with a `requires_positive_X` tag only accept
    # strictly positive data
    if tags.input_tags.positive_only:
        X = X - X.min()
        if X_test is not None:
            X_test = X_test - X_test.min()  # pragma: no cover
    if tags.input_tags.categorical:
        X = np.round(X - X.min())
        if X_test is not None:
            X_test = np.round(X_test - X_test.min())  # pragma: no cover
        if tags.input_tags.string:
            X = X.astype(object)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    X[i, j] = str(X[i, j])
            if X_test is not None:
                X_test = X_test.astype(object)
                for i in range(X_test.shape[0]):
                    for j in range(X_test.shape[1]):
                        X_test[i, j] = str(X_test[i, j])
        elif tags.input_tags.allow_nan:
            X = X.astype(np.float64)
            if X_test is not None:
                X_test = X_test.astype(np.float64)  # pragma: no cover
        else:
            X = X.astype(np.int32)
            if X_test is not None:
                X_test = X_test.astype(np.int32)  # pragma: no cover

    if estimator.__class__.__name__ == "SkewedChi2Sampler":
        # SkewedChi2Sampler requires X > -skewdness in transform
        X = X - X.min()
        if X_test is not None:
            X_test = X_test - X_test.min()  # pragma: no cover

    X_res = X

    # Pairwise estimators only accept
    # X of shape (`n_samples`, `n_samples`)
    if _is_pairwise_metric(estimator):
        X_res = pairwise_distances(X, metric="euclidean")
        if X_test is not None:
            X_test = pairwise_distances(
                X_test, X, metric="euclidean"
            )  # pragma: no cover
    elif tags.input_tags.pairwise:
        X_res = kernel(X, X)
        if X_test is not None:
            X_test = kernel(X_test, X)  # pragma: no cover
    if X_test is not None:
        return X_res, X_test
    return X_res


sklearn.utils.estimator_checks._enforce_estimator_tags_X = (
    _enforce_estimator_tags_X_monkey_patch
)


def _tested_estimators():
    for Estimator in [
        DatetimeEncoder,
        # Joiner,  # requires auxiliary tables
        GapEncoder,
        MinHashEncoder,
        SimilarityEncoder,
        TableVectorizer,
    ]:
        yield Estimator()


# TODO: remove the skip when the scikit-learn common test will be more lenient towards
# the string categorical data:
# xref: https://github.com/scikit-learn/scikit-learn/pull/26860
@pytest.mark.skip(
    "Common tests in scikit-learn are not allowing for categorical string data."
)
@parametrize_with_checks(list(_tested_estimators()))
def test_estimators_compatibility_sklearn(estimator, check, request):
    check(estimator)
