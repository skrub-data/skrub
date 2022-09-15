"""
dirty_cat: Learning on dirty categories.
"""
from ._check_dependencies import _version
from ._datetime_encoder import DatetimeEncoder
from ._gap_encoder import GapEncoder
from ._minhash_encoder import MinHashEncoder
from ._similarity_encoder import SimilarityEncoder
from ._super_vectorizer import SuperVectorizer
from ._target_encoder import TargetEncoder

__version__ = _version

__all__ = [
    "SimilarityEncoder",
    "TargetEncoder",
    "MinHashEncoder",
    "GapEncoder",
    "DatetimeEncoder",
    "SuperVectorizer",
]
