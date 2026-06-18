"""
Skrub DataOps for model definition, validation and hyper-parameter search.
==========================================================================

``DataOps`` are skrub's framework for building machine-learning pipelines. A
pipeline is first defined implicitly by performing operations on DataOp objects,
then encapsulated into a SkrubLearner which is a fittable estimator with a
scikit-learn-like interface (fit, predict, get_params etc.). The ``choose_*``
functions create special placeholders representing tunable hyperparameters that
can be inserted anywhere in the pipeline.
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
