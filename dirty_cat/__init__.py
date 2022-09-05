"""
dirty_cat: Learning on dirty categories.
"""
import os

from .datetime_encoder import DatetimeEncoder
from .gap_encoder import GapEncoder
from .minhash_encoder import MinHashEncoder
from .similarity_encoder import SimilarityEncoder
from .super_vectorizer import SuperVectorizer
from .target_encoder import TargetEncoder

version_file = os.path.join(os.path.dirname(__file__), "VERSION.txt")
with open(version_file) as fh:
    __version__ = fh.read().strip()

__all__ = [
    "SimilarityEncoder",
    "TargetEncoder",
    "MinHashEncoder",
    "GapEncoder",
    "DatetimeEncoder",
    "SuperVectorizer",
]
