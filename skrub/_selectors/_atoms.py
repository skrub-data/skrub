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
    def __init__(self, predicate, on_error="raise"):
        self.predicate = predicate
        allowed = ["raise", "reject", "accept"]
        if on_error not in allowed:
            raise ValueError(f"'on_error' must be one of {allowed}. Got {on_error!r}")
        self.on_error = on_error

    def select(self, df, ignore=()):
        cols = list_difference(sbd.column_names(df), ignore)
        result = []
        for col_name in cols:
            try:
                accept = self.predicate(sbd.col(df, col_name))
            except Exception:
                if self.on_error == "raise":
                    raise
                if self.on_error == "accept":
                    accept = True
                assert self.on_error == "reject"
                accept = False
            if accept:
                result.append(col_name)
        return result

    def __repr__(self):
        return f"filter({self.predicate!r})"


def filter(predicate, on_error="raise"):
    return Filter(predicate, on_error=on_error)


class FilterNames(Selector):
    def __init__(self, predicate):
        self.predicate = predicate

    def select(self, df, ignore=()):
        cols = list_difference(sbd.column_names(df), ignore)
        return [c for c in cols if self.predicate(c)]

    def __repr__(self):
        return f"filter_names({self.predicate!r})"


def filter_names(predicate):
    return FilterNames(predicate)


class CreatedBy(Selector):
    def __init__(self, *transformers):
        self.transformers = transformers

    def select(self, df, ignore=()):
        all_created = set()
        for step in self.transformers:
            if hasattr(step, "created_outputs_"):
                all_created.update(step.created_outputs_)
            else:
                all_created.update(step.get_feature_names_out())
        return list_intersect(sbd.column_names(df), all_created.difference(ignore))

    def __repr__(self):
        transformers_repr = f"<any of {len(self.transformers)} transformers>"
        return f"created_by({transformers_repr})"


def created_by(*transformers):
    return CreatedBy(*transformers)
