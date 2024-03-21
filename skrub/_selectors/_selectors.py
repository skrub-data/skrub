import fnmatch
import re

from .. import _dataframe as sbd
from ._base import Filter, NameFilter

__all__ = [
    "glob",
    "regex",
    "numeric",
    "integer",
    "float",
    "any_date",
    "categorical",
    "string",
    "boolean",
    "cardinality_below",
]

#
# Selectors based on column names
#


def glob(pattern):
    return NameFilter(fnmatch.fnmatch, args=(pattern,), name="glob")


def _regex(col_name, pattern):
    return re.match(pattern, col_name) is not None


def regex(pattern):
    return NameFilter(_regex, args=(pattern,), name="regex")


#
# Selectors based on data types
#


def numeric():
    return Filter(sbd.is_numeric, name="numeric")


def integer():
    return Filter(sbd.is_integer, name="integer")


def float():
    return Filter(sbd.is_float, name="float")


def any_date():
    return Filter(sbd.is_any_date, name="any_date")


def categorical():
    return Filter(sbd.is_categorical, name="categorical")


def string():
    return Filter(sbd.is_string, name="string")


def boolean():
    return Filter(sbd.is_bool, name="boolean")


#
# Selectors based on column values, computed statistics
#


def _cardinality_below(column, threshold):
    try:
        return sbd.n_unique(column) < threshold
    except Exception:
        # n_unique can fail for example for polars columns with dtype Object
        return False


def cardinality_below(threshold):
    return Filter(_cardinality_below, args=(threshold,), name="cardinality_below")
