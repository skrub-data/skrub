"""
skrub: Machine learning with dataframes.
=============================================

``skrub`` facilitates machine learning with tabular
data.  It helps clean, encode, and transform dataframes into features
ready for scikit-learn or other ML frameworks.

Bundled docs: ``skrub.__docs_dir__``
Bundled getting started: ``skrub.__docs_dir__ / "tutorials"``
Bundled examples: ``skrub.__docs_dir__ / "examples"``

Online docs: https://skrub-data.org/stable/reference/index.html
Source: https://github.com/skrub-data/skrub/
"""

from pathlib import Path as _Path

#: Path to the documentation bundled with the package.
#: Use ``skrub.__docs_dir__`` to access it programmatically.
__docs_dir__ = _Path(__file__).parent / "_docs"

from . import core, selectors
from ._agg_joiner import AggJoiner, AggTarget
from ._apply_to_cols import ApplyToCols
from ._column_associations import column_associations
from ._config import config_context, get_config, set_config
from ._data_ops import (
    DataOp,
    OptunaParamSearch,
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
from ._deduplicate import deduplicate
from ._drop_similar import DropSimilar
from ._drop_uninformative import DropUninformative
from ._duration_to_float import DurationToFloat
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
from ._tabular_pipeline import tabular_pipeline
from ._text_encoder import TextEncoder
from ._to_categorical import ToCategorical
from ._to_datetime import ToDatetime, to_datetime
from ._to_float import ToFloat

with open(_Path(__file__).parent / "VERSION.txt") as _fh:
    __version__ = _fh.read().strip()

__all__ = [
    "DataOp",
    "var",
    "SkrubLearner",
    "ParamSearch",
    "OptunaParamSearch",
    "X",
    "y",
    "as_data_op",
    "deferred",
    "eval_mode",
    "TableReport",
    "tabular_pipeline",
    "ApplyToCols",
    "DatetimeEncoder",
    "DurationToFloat",
    "ToDatetime",
    "ToFloat",
    "ToCategorical",
    "TableVectorizer",
    "TextEncoder",
    "StringEncoder",
    "Cleaner",
    "DropSimilar",
    "DropUninformative",
    "deduplicate",
    "to_datetime",
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
    "column_associations",
    "SquashingScaler",
    "patch_display",
    "unpatch_display",
    "get_config",
    "set_config",
    "GapEncoder",
    "MinHashEncoder",
    "SimilarityEncoder",
    "AggJoiner",
    "MultiAggJoiner",
    "AggTarget",
    "Joiner",
    "fuzzy_join",
    "InterpolationJoiner",
    "config_context",
    "core",
    "__docs_dir__",
]
