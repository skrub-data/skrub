"""
skrub: Prepping tables for machine learning.
"""
from pathlib import Path as _Path

from ._agg_joiner import AggJoiner, AggTarget
from ._check_dependencies import check_dependencies
from ._datetime_encoder import DatetimeEncoder, to_datetime
from ._deduplicate import compute_ngram_distance, deduplicate
from ._fuzzy_join import fuzzy_join
from ._gap_encoder import GapEncoder
from ._interpolation_joiner import InterpolationJoiner
from ._joiner import Joiner
from ._minhash_encoder import MinHashEncoder
from ._select_cols import DropCols, SelectCols
from ._similarity_encoder import SimilarityEncoder
from ._table_vectorizer import TableVectorizer

check_dependencies()

with open(_Path(__file__).parent / "VERSION.txt") as _fh:
    __version__ = _fh.read().strip()


__all__ = [
    "DatetimeEncoder",
    "Joiner",
    "fuzzy_join",
    "GapEncoder",
    "InterpolationJoiner",
    "MinHashEncoder",
    "SimilarityEncoder",
    "TableVectorizer",
    "deduplicate",
    "compute_ngram_distance",
    "to_datetime",
    "AggJoiner",
    "AggTarget",
    "SelectCols",
    "DropCols",
]
