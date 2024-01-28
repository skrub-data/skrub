import fnmatch
import re

from .. import _dataframe as sbd
from ._base import Selector
from ._utils import list_difference, list_intersect


class Glob(Selector):
    def __init__(self, pattern):
        self.pattern = pattern

    def select(self, df, ignore=()):
        df_cols = sbd.column_names(df)
        cols = set(df_cols).difference(ignore)
        selected = fnmatch.filter(cols, self.pattern)
        return list_intersect(df_cols, selected)

    def __repr__(self):
        return f"glob({self.pattern!r})"


def glob(pattern):
    return Glob(pattern)


class Regex(Selector):
    def __init__(self, pattern):
        self.pattern = pattern

    def select(self, df, ignore=()):
        pat = re.compile(self.pattern)
        cols = list_difference(sbd.column_names(df), ignore)
        return [c for c in cols if pat.match(c) is not None]

    def __repr__(self):
        return f"regex({self.pattern!r})"


def regex(pattern):
    return Regex(pattern)


class Filter(Selector):
    def __init__(self, predicate):
        self.predicate = predicate

    def select(self, df, ignore):
        cols = list_difference(sbd.column_names(df), ignore)
        return [c for c in cols if self.predicate(sbd.col(df, c))]

    def __repr__(self):
        return f"filter({self.predicate!r})"


def filter(predicate):
    return Filter(predicate)


class FilterNames(Selector):
    def __init__(self, predicate):
        self.predicate = predicate

    def select(self, df, ignore):
        cols = list_difference(sbd.column_names(df), ignore)
        return [c for c in cols if self.predicate(c)]

    def __repr__(self):
        return f"filter_names({self.predicate!r})"


def filter_names(predicate):
    return FilterNames(predicate)


class ProducedBy(Selector):
    def __init__(self, *transformers):
        self.transformers = transformers

    def select(self, df, ignore=()):
        all_produced = set()
        for step in self.transformers:
            if hasattr(step, "produced_outputs_"):
                all_produced.update(step.produced_outputs_)
            else:
                all_produced.update(step.get_feature_names_out())
        return list_intersect(sbd.column_names(df), all_produced.difference(ignore))

    def __repr__(self):
        transformers_repr = ", ".join(map(repr, self.transformers))
        return f"produced_by({transformers_repr})"


def produced_by(*transformers):
    return ProducedBy(*transformers)
