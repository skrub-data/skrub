"""
skrub: Prepping tables for machine learning.
"""

from pathlib import Path as _Path

from ._single_column_transformer import RejectColumn, SingleColumnTransformer

with open(_Path(__file__).parent / "VERSION.txt") as _fh:
    __version__ = _fh.read().strip()

__all__ = [
    "SingleColumnTransformer",
    "RejectColumn",
]
