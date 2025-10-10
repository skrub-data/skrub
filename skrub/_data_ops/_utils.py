import enum
import os
import shutil
import time
import traceback

from joblib.externals import cloudpickle

FITTED_PREDICTOR_METHODS = ("predict", "predict_proba", "decision_function", "score")
FITTED_ESTIMATOR_METHODS = FITTED_PREDICTOR_METHODS + ("transform",)
X_NAME = "_skrub_X"
Y_NAME = "_skrub_y"


class Sentinels(enum.Enum):
    NULL = "NULL"
    OPTIONAL_VALUE = "value"
    KFOLD_5 = "KFold(5)"

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


NULL = Sentinels.NULL
OPTIONAL_VALUE = Sentinels.OPTIONAL_VALUE
KFOLD_5 = Sentinels.KFOLD_5


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


def prune_folder(path: str):
    if not os.path.exists(path):
        return
    time_threshold = time.time() - 7 * 24 * 3600  # 7 days ago

    folders = (
        (f, os.path.getmtime(os.path.join(path, f)))
        for f in os.listdir(path)
        if os.path.isdir(os.path.join(path, f))
    )
    for dir in folders:
        try:
            if dir[1] < time_threshold and dir[0].startswith("full_data_op_report_"):
                shutil.rmtree(os.path.join(path, dir[0]))
        except Exception:
            pass
