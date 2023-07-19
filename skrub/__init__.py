"""
skrub: Prepping tables for machine learning.
"""
from pathlib import Path as _Path

try:
    from ._check_dependencies import check_dependencies

    check_dependencies()
except ModuleNotFoundError:
    import warnings

    warnings.warn(
        "packaging is not available, dependencies versions will not be checked."
    )

from ._datetime_encoder import DatetimeEncoder
from ._deduplicate import compute_ngram_distance, deduplicate
from ._feature_augmenter import FeatureAugmenter
from ._fuzzy_join import fuzzy_join
from ._gap_encoder import GapEncoder
from ._minhash_encoder import MinHashEncoder
from ._similarity_encoder import SimilarityEncoder
from ._table_vectorizer import SuperVectorizer, TableVectorizer

with open(_Path(__file__).parent / "VERSION.txt") as _fh:
    __version__ = _fh.read().strip()


__all__ = [
    "DatetimeEncoder",
    "FeatureAugmenter",
    "fuzzy_join",
    "GapEncoder",
    "MinHashEncoder",
    "SimilarityEncoder",
    "SuperVectorizer",
    "TableVectorizer",
    "deduplicate",
    "compute_ngram_distance",
]


def __getattr__(name):
    if name == "TargetEncoder":
        import warnings
        import sklearn
        from ._utils import parse_version

        if parse_version(sklearn.__version__) >= parse_version("1.4.0"):
            warnings.warn(
                "Importing TargetEncoder from skrub is deprecated and will be "
                "removed in a future version. It should be imported from sklearn."
            )
            from sklearn.preprocessing import TargetEncoder
        else:
            from ._target_encoder import TargetEncoder
        return TargetEncoder
    try:
        return globals()[name]
    except KeyError:
        # This is turned into the appropriate ImportError
        raise AttributeError
