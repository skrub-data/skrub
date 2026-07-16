import enum
import re
import shutil
import time
import traceback
import warnings
from pathlib import Path

from joblib.externals import cloudpickle

FITTED_PREDICTOR_METHODS = ("predict", "predict_proba", "decision_function", "score")
FITTED_ESTIMATOR_METHODS = FITTED_PREDICTOR_METHODS + ("transform",)
X_NAME = "_skrub_X"
Y_NAME = "_skrub_y"
IS_PREVIEW_DATA_ENV_NAME = "_skrub_is_preview_data_env"


class Sentinels(enum.Enum):
    NULL = "NULL"
    OPTIONAL_VALUE = "value"

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


NULL = Sentinels.NULL
OPTIONAL_VALUE = Sentinels.OPTIONAL_VALUE


def simple_repr(data_op):
    return repr(data_op).splitlines()[0].removeprefix("<").removesuffix(">")


def attribute_error(obj, name, comment=None):
    msg = f"{obj.__class__.__name__!r} object has no attribute {name!r}"
    if comment:
        msg = f"{msg}.\n{comment}"
    raise AttributeError(msg)


class _CloudPickle:
    def __getstate__(self):
        try:
            state = dict(super().__getstate__())
        except AttributeError:
            # before python 3.11
            state = self.__dict__.copy()
        for k in self._cloudpickle_attributes:
            state[k] = cloudpickle.dumps(state[k])
        return state

    def __setstate__(self, state):
        for k in self._cloudpickle_attributes:
            state[k] = cloudpickle.loads(state[k])
        object.__setattr__(self, "__dict__", state)


def format_exception(e):
    """compatibility for python < 3.10"""
    return traceback.format_exception(type(e), e, e.__traceback__)


def format_exception_only(e):
    """compatibility for python < 3.10"""
    return traceback.format_exception_only(type(e), e)


def prune_directory(path: str):
    path = Path(path)
    if not path.exists():
        return
    time_threshold = time.time() - 7 * 24 * 3600  # 7 days ago
    # pattern to match folder names like full_data_op_report_{datetime}_{randomstring}
    pattern = re.compile(r"^full_data_op_report_\d{4}-\d{2}-\d{2}T\d{6}_[0-9a-f]{8}$")
    for dir_path in path.iterdir():
        if (
            dir_path.is_dir()
            and pattern.match(dir_path.name)
            and dir_path.stat().st_mtime < time_threshold
        ):
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                warnings.warn(
                    "Skrub wants to delete an old folder in the skrub data folder: "
                    f"Could not delete {dir_path}:\n"
                    + "".join(format_exception_only(e))
                )


def unique_renaming():
    """Factory of unique names

    The returned function is called with a string and returns a string. If the
    input was seen before, the output will have a number appended so that all
    outputs are unique. This is best understood with an example (see below).

    Examples
    --------
    >>> from skrub._data_ops._utils import unique_renaming
    >>> rename = unique_renaming()
    >>> rename('a')
    'a'
    >>> rename('b')
    'b'
    >>> rename('a')
    'a_1'
    >>> rename('a')
    'a_2'
    >>> rename('c')
    'c'
    """
    used = set()

    def rename(name):
        if name not in used:
            used.add(name)
            return name
        i = 1
        while (numbered := f"{name}_{i}") in used:
            i += 1
        used.add(numbered)
        return numbered

    return rename


def graphviz_error_message(html=False):
    if html:
        return """\
To display the DataOp graph, please install Pydot and Graphviz
and make sure the dot command is in your <code>$PATH</code>.<br/>
You may also need to run <code>dot -c</code> in bash or powershell
to rebuild the plugin cache of Graphviz.<br/>
Graphviz must be installed using your system's
package manager rather than pip.<br/>
<a href="https://pypi.org/project/pydot/">Pydot documentation</a><br/>
<a href="https://graphviz.org/download/">Graphviz installation instructions</a><br/>
"""
    else:
        return """\
To display the DataOp graph,
please install Pydot and Graphviz and make sure the 'dot' command is in your $PATH.
You may also need to run 'dot -c' in bash or powershell
to rebuild the plugin cache of Graphviz.
Graphviz must be installed using your system's package manager rather than pip.
https://pypi.org/project/pydot/
https://graphviz.org/download/"""


def has_graphviz():
    try:
        import pydot

        g = pydot.Dot()
        g.add_node(pydot.Node("node 0"))
        g.create_svg()
        return True
    except Exception:
        return False


def check_graphviz():
    if has_graphviz():
        return
    raise RuntimeError(graphviz_error_message())
