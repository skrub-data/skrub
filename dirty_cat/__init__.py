"""
dirty_cat: Learning on dirty categories.
"""
from pathlib import Path as _Path

from ._datetime_encoder import DatetimeEncoder
from ._gap_encoder import GapEncoder
from ._minhash_encoder import MinHashEncoder
from ._similarity_encoder import SimilarityEncoder
from ._super_vectorizer import SuperVectorizer
from ._target_encoder import TargetEncoder

with open(_Path(__file__).parent / "_VERSION.txt") as _fh:
    __version__ = _fh.read().strip()

__all__ = [
    "SimilarityEncoder",
    "TargetEncoder",
    "MinHashEncoder",
    "GapEncoder",
    "DatetimeEncoder",
    "SuperVectorizer",
]
