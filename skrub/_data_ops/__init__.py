"""
Skrub DataOps for model creation, estimation and hyper-parameter search.
=========================================================================

``DataOps`` are skrub's declarative pipeline abstraction.  A DataOp
represents a computation — from data loading and column selection to
scikit-learn estimator application — that can be inspected, previewed,
exported, and re-used.
"""

from ._choosing import (
    choose_bool,
    choose_float,
    choose_from,
    choose_int,
    optional,
)
from ._data_ops import DataOp, X, as_data_op, deferred, eval_mode, var, y
from ._estimator import ParamSearch, SkrubLearner, cross_validate
from ._optuna import OptunaParamSearch

__all__ = [
    "DataOp",
    "var",
    "X",
    "y",
    "as_data_op",
    "deferred",
    "eval_mode",
    "cross_validate",
    "SkrubLearner",
    "ParamSearch",
    "OptunaParamSearch",
    #
    "choose_bool",
    "choose_float",
    "choose_from",
    "choose_int",
    "optional",
]
