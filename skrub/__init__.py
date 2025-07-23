"""
skrub: Prepping tables for machine learning.
"""

from pathlib import Path as _Path

from . import selectors
from ._agg_joiner import AggJoiner, AggTarget
from ._apply_to_cols import ApplyToCols
from ._apply_to_frame import ApplyToFrame
from ._column_associations import column_associations
from ._config import config_context, get_config, set_config
from ._data_ops import (
    DataOp,
    ParamSearch,
    SkrubLearner,
    X,
    as_data_op,
    choose_bool,
    choose_float,
    choose_from,
    choose_int,
    cross_validate,
    deferred,
    eval_mode,
    optional,
    var,
    y,
)
from ._datetime_encoder import DatetimeEncoder
from ._deduplicate import compute_ngram_distance, deduplicate
from ._drop_uninformative import DropUninformative
from ._fuzzy_join import fuzzy_join
from ._gap_encoder import GapEncoder
from ._interpolation_joiner import InterpolationJoiner
from ._joiner import Joiner
from ._minhash_encoder import MinHashEncoder
from ._multi_agg_joiner import MultiAggJoiner
from ._reporting import TableReport, patch_display, unpatch_display
from ._select_cols import Drop, DropCols, SelectCols
from ._similarity_encoder import SimilarityEncoder
from ._squashing_scaler import SquashingScaler
from ._string_encoder import StringEncoder
from ._table_vectorizer import Cleaner, TableVectorizer
from ._tabular_pipeline import tabular_learner, tabular_pipeline
from ._text_encoder import TextEncoder
from ._to_categorical import ToCategorical
from ._to_datetime import ToDatetime, to_datetime

with open(_Path(__file__).parent / "VERSION.txt") as _fh:
    __version__ = _fh.read().strip()

__all__ = [
    "DataOp",
    "var",
    "SkrubLearner",
    "ParamSearch",
    "X",
    "y",
    "as_data_op",
    "deferred",
    "eval_mode",
    "TableReport",
    "patch_display",
    "unpatch_display",
    "tabular_pipeline",
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
    "DropUninformative",
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
    "cross_validate",
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
    "SquashingScaler",
    "get_config",
    "set_config",
    "config_context",
    "ApplyToCols",
    "ApplyToFrame",
]
