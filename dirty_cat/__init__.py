"""
dirty_cat: Learning on dirty categories.
"""
from pathlib import Path as _Path

from ._check_dependencies import check_dependencies

check_dependencies()

from ._datetime_encoder import DatetimeEncoder  # noqa: E402
from ._gap_encoder import GapEncoder  # noqa: E402
from ._minhash_encoder import MinHashEncoder  # noqa: E402
from ._similarity_encoder import SimilarityEncoder  # noqa: E402
from ._super_vectorizer import SuperVectorizer  # noqa: E402
from ._target_encoder import TargetEncoder  # noqa: E402

parent_dir = _Path(__file__).parent
with open(parent_dir / "VERSION.txt") as _fh:
    __version__ = _fh.read().strip()


__all__ = [
    "SimilarityEncoder",
    "TargetEncoder",
    "MinHashEncoder",
    "GapEncoder",
    "DatetimeEncoder",
    "SuperVectorizer",
]
