from .. import _dataframe as sbd
from . import _expressions

SHOULD_SUBSAMPLE_KEY = "_skrub_should_subsample"


def _should_subsample(mode, environment):
    if mode == "preview":
        return True
    if "fit" not in mode:
        return False
    return environment.get(SHOULD_SUBSAMPLE_KEY, False)


class _ShouldSubsample(_expressions.ExprImpl):
    _fields = []

    def compute(self, e, mode, environment):
        return _should_subsample(mode, environment)


@_expressions.check_expr
def should_subsample():
    """Expression indicating if subsampling should be applied.

    This is a helper for other skrub functions that need the information of
    whether they should apply any subsampling, depending on the current
    evaluation mode and parameters passed by the user.

    Examples
    --------
    >>> import skrub
    >>> from skrub._expressions._subsampling import should_subsample

    >>> @skrub.deferred
    ... def load_data(subsampling=should_subsample()):
    ...     print("subsampling:", subsampling)
    ...     return 1.0 if subsampling else 2.0

    >>> e = load_data()
    subsampling: True
    >>> e
    <Call 'load_data'>
    Result:
    ―――――――
    1.0
    >>> e.skb.get_estimator(fitted=True)
    subsampling: False
    ExprEstimator(expr=<Call 'load_data'>)
    >>> e.skb.get_estimator(subsampling=True, fitted=True)
    subsampling: True
    ExprEstimator(expr=<Call 'load_data'>)
    """
    return _expressions.Expr(_ShouldSubsample())


class PreviewSubsample(_expressions.ExprImpl):
    """Optionally subsample a dataframe.

    See the docstring of ``.skb.preview_subsample`` for details.
    """

    _fields = ["df", "n", "how"]

    def compute(self, e, mode, environment):
        if e.how not in ["head", "random"]:
            raise ValueError("`how` should be 'head' or 'random', got: {e.how!r}")
        if not sbd.is_dataframe(e.df):
            # TODO sampling columns, numpy arrays
            raise TypeError("To use subsampling, `df` should be a dataframe.")
        if not _should_subsample(mode, environment):
            return e.df
        if e.how == "head":
            return sbd.head(e.df, n=e.n)
        return sbd.sample(e.df, n=e.n, seed=0)


def env_with_subsampling(environment, subsampling):
    """Update an environment with subsampling indication.

    Small private helper to add subsampling to an environment, if subsampling
    is required. If `subsampling` is False, the environment is left as-is. That
    is because subsampling might have been turned on at a higher level (e.g. a
    cross-validation loop), in which case we don't turn it off at the lower
    level (e.g. fitting an estimator).
    """
    if not subsampling:
        return environment
    return environment | {SHOULD_SUBSAMPLE_KEY: True}
