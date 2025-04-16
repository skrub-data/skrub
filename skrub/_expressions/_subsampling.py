from .. import _dataframe as sbd
from . import _expressions

SHOULD_SUBSAMPLE_KEY = "_skrub_should_subsample"


def _should_subsample(mode, environment):
    if mode == "preview":
        return True
    if "fit" not in mode:
        return False
    return environment.get(SHOULD_SUBSAMPLE_KEY, False)


@_expressions.deferred
def should_subsample(
    mode=_expressions.eval_mode(), environment=_expressions.eval_environment()
):
    return _should_subsample(mode, environment)


class PreviewSubsample(_expressions.ExprImpl):
    _fields = ["df", "n", "how"]

    def compute(self, e, mode, environment):
        if e.how not in ["head", "random"]:
            raise ValueError("`how` should be 'head' or 'random', got: {e.how!r}")
        if not sbd.is_dataframe(e.df):
            # TODO sampling columns, numpy arrays
            raise TypeError("`df` should be a dataframe.")
        if not _should_subsample(mode, environment):
            return e.df
        if e.how == "head":
            return sbd.head(e.df, n=e.n)
        return sbd.sample(e.df, n=e.n, seed=0)
