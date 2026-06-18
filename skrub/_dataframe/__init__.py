"""
Internal dataframe abstraction layer.
======================================

Provides a single-point-of-dispatch API over pandas and polars.
This module is considered **private**; users should not rely on its
contents directly.
"""

from . import _common
from ._common import *  # noqa: F403

__all__ = _common.__all__
