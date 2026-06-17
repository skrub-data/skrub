"""
Data operations (``DataOp``), estimation and hyper-parameter search.
====================================================================

``DataOp`` is skrub's declarative pipeline abstraction.  A DataOp
represents a computation — from data loading and column selection to
scikit-learn estimator application — that can be inspected, previewed,
exported, and re-used without executing until :meth:`~skrub.DataOp.skb.eval`
is called.

This subpackage provides:

- :class:`~skrub.DataOp` — the core declarative computation object
  (its fluent API is accessed via the ``.skb`` accessor).
- :class:`~skrub.SkrubLearner` — a scikit-learn-compatible learner wrapper
  that evaluates a DataOp.
- :class:`~skrub.ParamSearch` and :class:`~skrub.OptunaParamSearch` —
  grid / optuna-based hyper-parameter optimisation.
- Helper functions: :func:`~skrub.cross_validate`,
  :func:`~skrub.deferred`, :func:`~skrub.eval_mode`, and the
  ``X`` / ``y`` / ``var`` placeholders.
- Choosing utilities: :func:`~skrub.choose_bool`,
  :func:`~skrub.choose_float`, :func:`~skrub.choose_from`,
  :func:`~skrub.choose_int`, :func:`~skrub.optional`.

Anything not listed in ``__all__`` is private and should not be used
directly.
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
