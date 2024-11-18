import sklearn
from sklearn.utils.fixes import parse_version

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)


if sklearn_version < parse_version("1.6"):
    from sklearn.utils._tags import _safe_tags as get_tags  # noqa
else:
    from sklearn.utils import get_tags  # noqa


def _check_n_features(estimator, X, *, reset):
    if hasattr(estimator, "_check_n_features"):
        estimator._check_n_features(X, reset=reset)
    else:
        from sklearn.utils.validation import _check_n_features

        _check_n_features(estimator, X, reset=reset)
