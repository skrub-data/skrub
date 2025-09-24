import numpy as np

from .. import _dataframe as sbd
from .._config import get_config
from . import _data_ops, _evaluation

# The key in the evaluation environment that indicates if subsampling should
# take place or not. Subsampling can be turned on or off when evaluating a
# DataOp by setting the corresponding value in the environment.

SHOULD_SUBSAMPLE_KEY = "_skrub_should_subsample"


def _should_subsample(mode, environment):
    enable_subsampling = get_config()["enable_subsampling"]
    if "fit" not in mode and mode != "preview":
        # We never want to subsample outside of fit, fit_transform and preview.
        return False
    if enable_subsampling == "disable":
        return False
    if mode == "preview":
        return True
    if enable_subsampling == "force":
        return True
    return environment.get(SHOULD_SUBSAMPLE_KEY, False)


class ShouldSubsample(_data_ops.DataOpImpl):
    _fields = []

    def compute(self, e, mode, environment):
        return _should_subsample(mode, environment)


@_data_ops.check_data_op
def should_subsample():
    """DataOp indicating if subsampling should be applied.

    This is a helper for other skrub functions that need the information of
    whether they should apply any subsampling, depending on the current
    evaluation mode and parameters passed by the user.

    Examples
    --------
    >>> import skrub
    >>> from skrub._data_ops._subsampling import should_subsample

    >>> @skrub.deferred
    ... def load_data(subsampling=should_subsample()):
    ...     print("subsampling:", subsampling)
    ...     return 1.0 if subsampling else 2.0

    >>> e = load_data()
    subsampling: True
    >>> e
    <Call 'load_data'>
    Result (on a subsample):
    ――――――――――――――――――――――――
    1.0
    >>> e.skb.make_learner(fitted=True)
    subsampling: False
    SkrubLearner(data_op=<Call 'load_data'>)
    >>> e.skb.make_learner(keep_subsampling=True, fitted=True)
    subsampling: True
    SkrubLearner(data_op=<Call 'load_data'>)
    """
    return _data_ops.DataOp(ShouldSubsample())


def _sample_numpy(a, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(a.shape[0], size=n, replace=False)
    return a[idx]


def _head_numpy(a, n):
    return a[:n]


class SubsamplePreviews(_data_ops.DataOpImpl):
    """Optionally subsample a dataframe.

    See the docstring of ``.skb.subsample`` for details.
    """

    _fields = ["target", "n", "how"]

    def compute(self, e, mode, environment):
        if e.how not in ["head", "random"]:
            raise ValueError("`how` should be 'head' or 'random', got: {e.how!r}")
        is_numpy = isinstance(e.target, np.ndarray)
        if not (is_numpy or sbd.is_dataframe(e.target) or sbd.is_column(e.target)):
            raise TypeError(
                "To use subsampling, the input should be a dataframe, "
                f"column or numpy array, got an object of type: {type(e.target)}."
            )
        if not _should_subsample(mode, environment):
            return e.target
        shape = e.target.shape if is_numpy else sbd.shape(e.target)
        n = min(e.n, shape[0])
        if e.how == "head":
            return _head_numpy(e.target, n=n) if is_numpy else sbd.head(e.target, n=n)
        seed = get_config()["subsampling_seed"]
        return (
            _sample_numpy(e.target, n=n, seed=seed)
            if is_numpy
            else sbd.sample(e.target, n=n, seed=seed)
        )


def env_with_subsampling(data_op, environment, keep_subsampling):
    """Update an environment with subsampling indication.

    Small private helper to add subsampling to an environment, if subsampling
    is required. If `keep_subsampling` is False, the environment is left as-is. That
    is because subsampling might have been turned on at a higher level (e.g. a
    cross-validation loop), in which case we don't turn it off at the lower
    level (e.g. fitting an estimator).
    """
    if not keep_subsampling:
        return environment
    if not uses_subsampling(data_op):
        raise ValueError(
            "`keep_subsampling=True` was passed but no subsampling has been configured"
            " anywhere in the DataOp. Either pass `keep_subsampling=False` (the"
            " default) or configure subsampling with `.skb.subsample()`."
        )
    return environment | {SHOULD_SUBSAMPLE_KEY: True}


def uses_subsampling(data_op):
    """Find if subsampling is configured somewhere in the DataOp.

    This can be used for example to notify the user that the preview they see
    comes from a subsample.

    Note that the check is a bit too simple so there may be false positives
    (this returns True but actually all of the data was used) for example if
    subsampling was done with ``subsample size >= data size`` and
    ``how='head'``, or if subsampling takes place in a path that was not used
    for the preview (e.g. the unused branch of a ``.skb.if_else()``
    DataOp).
    """
    return (
        _evaluation.find_node(
            data_op,
            lambda e: isinstance(e, _data_ops.DataOp)
            and isinstance(e._skrub_impl, (SubsamplePreviews, ShouldSubsample)),
        )
        is not None
    )
