"""
skrub: Prepping tables for machine learning.
=============================================

``skrub`` (formerly *dirty_cat*) facilitates machine learning with tabular
data.  It helps clean, encode, join, and transform dataframes into features
ready for scikit-learn or other ML frameworks.

Docs: https://skrub-data.org/stable/reference/index.html
User Guide: https://skrub-data.org/stable/documentation.html
Source: https://github.com/skrub-data/skrub/
Examples: https://skrub-data.org/stable/auto_examples/index.html

The public API follows the User Guide structure described below.

Default wrangling (building a quick pipeline)
----------------------------------------------
- :class:`~skrub.Cleaner` — sanitise dtypes, nulls and dates.
- :class:`~skrub.TableVectorizer` — one-step dataframe-to-numeric conversion.
- :func:`~skrub.tabular_pipeline` — assemble a trainable pipeline with
  sensible defaults.
- :class:`~skrub.ApplyToCols` — wrap any transformer for selected columns.

Column-level feature extraction
-------------------------------
- **String / text encoders**: :class:`~skrub.GapEncoder`,
  :class:`~skrub.MinHashEncoder`, :class:`~skrub.SimilarityEncoder`,
  :class:`~skrub.StringEncoder`, :class:`~skrub.TextEncoder`
- **Datetime handling**: :class:`~skrub.DatetimeEncoder`,
  :class:`~skrub.ToDatetime`, :func:`~skrub.to_datetime`
- **Numerical / scaling**: :class:`~skrub.SquashingScaler`,
  :class:`~skrub.ToFloat`, :class:`~skrub.DurationToFloat`
- **Categorical**: :class:`~skrub.ToCategorical`

Multi-column operations
-----------------------
- **Selectors**: :mod:`skrub.selectors` — filter columns by dtype, name,
  cardinality, etc.  Also :class:`~skrub.Drop`,
  :class:`~skrub.DropCols`, :class:`~skrub.SelectCols`.
- :class:`~skrub.DropUninformative`, :class:`~skrub.DropSimilar`

Joining dataframes
------------------
Joiners are available but have various shortcomings so they are not recommended
for general use. Use the Data Ops instead.
:class:`~skrub.Joiner`, :func:`~skrub.fuzzy_join`,
:class:`~skrub.AggJoiner`, :class:`~skrub.MultiAggJoiner`,
:class:`~skrub.AggTarget`, :class:`~skrub.InterpolationJoiner`

DataOps (declarative ML pipelines)
-----------------------------------
:class:`~skrub.DataOp`, :class:`~skrub.SkrubLearner`,
:class:`~skrub.ParamSearch`, :class:`~skrub.OptunaParamSearch`,
:func:`~skrub.cross_validate`, the ``X`` / ``y`` / ``var`` placeholders,
:func:`~skrub.deferred`, :func:`~skrub.eval_mode`, and the
:func:`~skrub.choose_bool` / :func:`~skrub.choose_float` /
:func:`~skrub.choose_from` / :func:`~skrub.choose_int` /
:func:`~skrub.optional` helpers for hyper-parameter tuning.

Exploring a dataframe
---------------------
:class:`~skrub.TableReport`, :func:`~skrub.patch_display`,
:func:`~skrub.unpatch_display`

Utilities
---------
:func:`~skrub.deduplicate`, :func:`~skrub.column_associations`,
:func:`~skrub.set_config`, :func:`~skrub.get_config`,
:func:`~skrub.config_context`

Subpackages
-----------
- :mod:`skrub.selectors` — column selection primitives.
- :mod:`skrub._data_ops` — DataOp internals (re-exported above).
- :mod:`skrub.datasets` — fetching real-world datasets and generating
  synthetic toy data.
- ``skrub.core`` — base classes ``SingleColumnTransformer`` and
  ``RejectColumn`` (mostly for internal / advanced use).

Privacy note
------------
Any module or function that is not imported in this ``__init__.py`` is
considered **private** and should not be used directly.  Private modules
are prefixed with an underscore (e.g. ``_dataframe``, ``_table_vectorizer``).
"""

from pathlib import Path as _Path

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
]
