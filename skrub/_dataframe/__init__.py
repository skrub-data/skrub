"""
Internal dataframe abstraction layer.
======================================

This module provides a single-point-of-dispatch API over pandas and polars, to
allow skrub to work with either dataframe library.  It is used internally by
skrub's dataframe transformers and reporting tools.
This module is considered **private**; users should not rely on its
contents directly.
"""

from . import _common
from ._common import *  # noqa: F403

__all__ = _common.__all__
