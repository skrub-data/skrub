"""
Internal dataframe abstraction layer.
======================================

Provides a single-point-of-dispatch API over pandas and polars.  Functions
such as :func:`is_numeric`, :func:`to_datetime`, :func:`concat`,
:func:`col`, :func:`name`, :func:`dtype`, :func:`has_nulls`, etc. all
accept a dataframe or column from any supported backend and dispatch to the
correct implementation automatically.

This module is considered **private**; users should not rely on its
contents directly.  See ``__all__`` for the list of exported helpers.

>>> from skrub import _dataframe as sbd
>>> sbd.is_numeric(pd.Series([1, 2, 3]))
True
"""

from . import _common
from ._common import *  # noqa: F403

__all__ = _common.__all__
