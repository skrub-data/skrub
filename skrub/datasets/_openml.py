import sklearn
from sklearn.datasets import fetch_openml
from sklearn.utils.fixes import parse_version

from ._utils import get_data_dir

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)


def fetch_openml_skb(
    data_id,
    data_home=None,
    target_column="default-target",
    cache=True,
    return_X_y=False,
    as_frame=True,
    n_retries=3,
    delay=1,
    parser=None,
):
    """Downloads a dataset from OpenML, taking care of creating the directories.

    The ``fetch_openml()`` function returns a scikit-learn ``Bunch`` object,
    which behaves just like a ``namedtuple``.
    However, we do not want to save this data into memory: we will read it from
    the disk later.
    """
    data_home = data_home or get_data_dir(name="openml")
    if parser is None:
        # Avoid the warning, but don't use auto yet because of
        # https://github.com/scikit-learn/scikit-learn/issues/25478
        if parse_version("1.2") <= sklearn_version < parse_version("1.2.2"):
            parser = "liac-arff"
        else:
            parser = "auto"
    fetch_openml(
        data_id=data_id,
        data_home=data_home,
        target_column=target_column,
        cache=cache,
        return_X_y=return_X_y,
        as_frame=as_frame,
        n_retries=n_retries,
        delay=delay,
        parser=parser,
    )
