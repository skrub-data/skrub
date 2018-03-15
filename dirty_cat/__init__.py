"""
dirty_cat: Learning on dirty categories.
"""
import sys

__version__ = '0.0.1a'

try:
    __DIRTY_CAT_SETUP__
except NameError:
    __DIRTY_CAT_SETUP__ = False

if __DIRTY_CAT_SETUP__:
    sys.stderr.write('Partial import of dirty_cat during installation.\n')
else:
    from .similarity_encoder import SimilarityEncoder
    from .target_encoder import TargetEncoder

    __all__ = ['SimilarityEncoder', 'TargetEncoder']
