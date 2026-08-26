"""
Core base classes for skrub transformers.
==========================================

This module provides :class:`SingleColumnTransformer` and
:class:`RejectColumn`, which are used as building blocks for skrub's
column-level transformers.

These classes are re-exported from ``skrub.core`` for advanced use-cases;
most users should not need them directly.
"""

from ._single_column_transformer import RejectColumn, SingleColumnTransformer

__all__ = ["RejectColumn", "SingleColumnTransformer"]
