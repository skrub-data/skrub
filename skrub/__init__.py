"""
skrub: Prepping tables for machine learning.
"""

from pathlib import Path as _Path

from . import _selectors as selectors
from ._agg_joiner import AggJoiner, AggTarget
from ._column_associations import column_associations
from ._datetime_encoder import DatetimeEncoder
from ._deduplicate import compute_ngram_distance, deduplicate
from ._expressions import (
    Expr,
    ExprEstimator,
    ParamSearch,
    X,
    as_expr,
    choose_bool,
    choose_float,
    choose_from,
    choose_int,
    cross_validate,
    deferred,
    eval_mode,
    optional,
    train_test_split,
    var,
    y,
)
from ._fuzzy_join import fuzzy_join
from ._gap_encoder import GapEncoder
from ._interpolation_joiner import InterpolationJoiner
from ._joiner import Joiner
from ._minhash_encoder import MinHashEncoder
from ._multi_agg_joiner import MultiAggJoiner
from ._reporting import TableReport, patch_display, unpatch_display
from ._select_cols import Drop, DropCols, SelectCols
from ._similarity_encoder import SimilarityEncoder
from ._string_encoder import StringEncoder
from ._table_vectorizer import Cleaner, TableVectorizer
from ._tabular_learner import tabular_learner
from ._text_encoder import TextEncoder
from ._to_categorical import ToCategorical
from ._to_datetime import ToDatetime, to_datetime
from .datasets import toy_orders

with open(_Path(__file__).parent / "VERSION.txt") as _fh:
    __version__ = _fh.read().strip()


__all__ = [
    "Expr",
    "var",
    "ExprEstimator",
    "ParamSearch",
    "X",
    "y",
    "as_expr",
    "deferred",
    "eval_mode",
    "TableReport",
    "patch_display",
    "unpatch_display",
    "tabular_learner",
    "DatetimeEncoder",
    "ToDatetime",
    "Joiner",
    "fuzzy_join",
    "GapEncoder",
    "InterpolationJoiner",
    "MinHashEncoder",
    "SimilarityEncoder",
    "TableVectorizer",
    "Cleaner",
    "deduplicate",
    "compute_ngram_distance",
    "ToCategorical",
    "to_datetime",
    "AggJoiner",
    "MultiAggJoiner",
    "AggTarget",
    "SelectCols",
    "DropCols",
    "Drop",
    "Recipe",
    "cross_validate",
    "train_test_split",
    "choose_from",
    "optional",
    "choose_float",
    "choose_int",
    "choose_bool",
    "selectors",
    "TextEncoder",
    "StringEncoder",
    "column_associations",
    "toy_orders",
]
