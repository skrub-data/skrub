"""
Utilities for manipulating dataframes
=====================================

This private module provides a collection of functions for manipulating
dataframes and series, such as ``mean()`` or ``select()``, that work on both
Pandas and Polars inputs (by relying on the mechanism provided by
``_dispatch.dispatch``).

Other skrub modules should rely on those facilities for all dataframe
operations. Other skrub modules can also use ``dispatch`` directly to define
their own generic functions.

This module is private and its functionality is not exposed to skrub users.
"""

from . import _common
from ._common import *  # noqa: F403

__all__ = _common.__all__
