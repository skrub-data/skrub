import collections
from typing import Any, Hashable, Tuple, Union

import numpy as np
from sklearn.utils import check_array


class LRUDict:
    """dict with limited capacity

    Using LRU eviction avoids memorizing a full dataset"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def __getitem__(self, key: Hashable):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return -1

    def __setitem__(self, key: Hashable, value: Any):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def __contains__(self, key: Hashable):
        return key in self.cache


def check_input(X) -> np.array:
    """
    Check input with sklearn standards.
    Also converts X to a numpy array if not already.
    """
    # TODO check for weird type of input to pass scikit learn tests
    #  without messing with the original type too much

    X_ = check_array(
        X,
        dtype=None,
        ensure_2d=True,
        force_all_finite=False,
    )
    # If the array contains both NaNs and strings, convert to object type
    if X_.dtype.kind in {"U", "S"}:  # contains strings
        if np.any(X_ == "nan"):  # missing value converted to string
            return check_array(
                np.array(X, dtype=object),
                dtype=None,
                ensure_2d=True,
                force_all_finite=False,
            )

    return X_


class Version:
    """
    Replacement for `distutil.version.LooseVersion` and
    `packaging.version.Version`.
    Implemented to avoid `DeprecationWarning`s raised by the former,
    and avoid adding a dependency for the latter.

    It is therefore very bare-bones, so its code shouldn't be too
    hard to understand.
    It currently only supports major and minor versions.

    Inspired from https://stackoverflow.com/a/11887825/9084059
    Should eventually dissapear.

    Examples:
    >>> # Standard usage
    >>> Version(sklearn.__version__) > Version('0.23')
    >>> Version(sklearn.__version__) > '0.23'
    >>> # In general, pass the version as numbers separated by dots.
    >>> Version('1.5') <= Version('1.6.5')
    >>> Version('1.5') <= '1.6.5'
    >>> # You can also pass the separator for specific cases
    >>> Version('1-5', separator='-') == Version('1-6-5', separator='-')
    >>> Version('1-5', separator='-') == '1-6-5'
    >>> Version('1-5', separator='-') == '1.6.5'  # Won't work!
    """

    def __init__(self, value: str, separator: str = "."):
        self.separator = separator
        self.major, self.minor = self._parse_version(value)

    def __repr__(self):
        return f"Version({self.major}.{self.minor})"

    def _parse_version(self, value: str) -> Tuple[int, int]:
        raw_parts = value.split(self.separator)
        if len(raw_parts) == 0:
            raise ValueError(
                f"Could not extract version from {value!r} "
                f"(separator: {self.separator!r})"
            )
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
        return (self.major, self.minor) == (other.major, other.minor)

    def __ne__(self, other: Union["Version", str]):
        other = self._cast_to_version(other)
        return (self.major, self.minor) != (other.major, other.minor)

    def __lt__(self, other: Union["Version", str]):
        other = self._cast_to_version(other)
        return (self.major, self.minor) < (other.major, other.minor)

    def __le__(self, other: Union["Version", str]):
        other = self._cast_to_version(other)
        return (self.major, self.minor) <= (other.major, other.minor)

    def __gt__(self, other: Union["Version", str]):
        other = self._cast_to_version(other)
        return (self.major, self.minor) > (other.major, other.minor)

    def __ge__(self, other: Union["Version", str]):
        other = self._cast_to_version(other)
        return (self.major, self.minor) >= (other.major, other.minor)
