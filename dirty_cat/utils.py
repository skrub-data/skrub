import collections

import numpy as np

from typing import Tuple, Union


class LRUDict:
    """ dict with limited capacity

    Using LRU eviction, this avoid to memorizz a full dataset"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def __getitem__(self, key):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return -1

    def __setitem__(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache


def check_input(X):
    """
    Check input data shape.
    Also converts X to a numpy array if not already.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(
            'Expected 2D array. Reshape your data either using'
            'array.reshape(-1, 1) if your data has a single feature or'
            'array.reshape(1, -1) if it contains a single sample.'
        )
    return X


class Version:
    """
    Replacement for `distutil.version.LooseVersion` and `packaging.version.Version`.
    Implemented to avoid `DeprecationWarning`s raised by the former,
    and avoid adding a dependency for the latter.

    It is therefore very bare-bones, so its code shouldn't be too
    hard to understand.
    It currently only supports major and minor versions.

    Examples:
    >>> # Standard usage
    >>> Version(sklearn.__version__) > Version('0.22')
    >>> Version(sklearn.__version__) > '0.22'
    >>> # In general, pass the version as numbers separated by dots.
    >>> Version('1.5') <= Version('1.6.5')
    >>> Version('1.5') <= '1.6.5'
    >>> # You can also pass the separator for specific cases
    >>> Version('1-5', separator='-') == Version('1-6-5', separator='-')
    >>> Version('1-5', separator='-') == '1-6-5'
    >>> Version('1-5', separator='-') == '1.6.5'  # Won't work !
    """

    def __init__(self, value: str, separator: str = '.'):
        self.separator = separator
        self.major, self.minor = self._parse_version(value)

    def _parse_version(self, value: str) -> Tuple[int, int]:
        raw_parts = value.split(self.separator)
        if len(raw_parts) == 0:
            raise ValueError(f'Could not extract version from {value!r} '
                             f'(separator: {self.separator!r})')
        elif len(raw_parts) == 1:
            major = int(raw_parts[0])
            minor = 0
        else:
            major = int(raw_parts[0])
            minor = int(raw_parts[1])
            # Ditch the rest
        return major, minor

    def _cast_to_version(self, other: Union["Version", str]) -> "Version":
        if isinstance(other, str):
            # We pass our separator, as we expect they are the same
            other = Version(other, self.separator)
        return other

    def __eq__(self, other: Union["Version", str]):
        other = self._cast_to_version(other)
        return (self.major == other.major) and (self.minor == other.minor)

    def __ne__(self, other: Union["Version", str]):
        other = self._cast_to_version(other)
        return (self.major != other.major) and (self.minor != other.minor)

    def __lt__(self, other: Union["Version", str]):
        other = self._cast_to_version(other)
        return (self.major < other.major) and (self.minor < other.minor)

    def __le__(self, other: Union["Version", str]):
        other = self._cast_to_version(other)
        return (self.major <= other.major) and (self.minor <= other.minor)

    def __gt__(self, other: Union["Version", str]):
        other = self._cast_to_version(other)
        return (self.major > other.major) and (self.minor > other.minor)

    def __ge__(self, other: Union["Version", str]):
        other = self._cast_to_version(other)
        return (self.major >= other.major) and (self.minor >= other.minor)
