"""
dirty_cat: Learning on dirty categories.
"""
from .check_dependencies import _version
from .datetime_encoder import DatetimeEncoder
from .gap_encoder import GapEncoder
from .minhash_encoder import MinHashEncoder
from .similarity_encoder import SimilarityEncoder
from .super_vectorizer import SuperVectorizer
from .target_encoder import TargetEncoder

__version__ = _version
__all__ = [
    "SimilarityEncoder",
    "TargetEncoder",
    "MinHashEncoder",
    "GapEncoder",
    "DatetimeEncoder",
    "SuperVectorizer",
]
