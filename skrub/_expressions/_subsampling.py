from .. import _dataframe as sbd
from . import _expressions

SHOULD_SUBSAMPLE_KEY = "_skrub_should_subsample"


@_expressions.deferred
def should_subsample(
    mode=_expressions.eval_mode(), env=_expressions.eval_environment()
):
    if mode == "preview":
        return True
    return env.get("_skrub_should_subsample", False)


@_expressions.deferred
def preview_subsample(df, n, how="head", should_subsample=should_subsample()):
    if how not in ["head", "random"]:
        raise ValueError("`how` should be 'head' or 'random', got: {how!r}")
    if not should_subsample:
        return df
    if how == "head":
        return sbd.head(df, n=n)
    return sbd.sample(df, n=n, seed=0)
