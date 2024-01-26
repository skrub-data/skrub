import fnmatch
import re

from ._base import Selector


class Glob(Selector):
    def __init__(self, pattern):
        self.pattern = pattern

    def select(self, df):
        return fnmatch.filter(df.columns, self.pattern)

    def __repr__(self):
        return f"glob({self.pattern!r})"


def glob(pattern):
    return Glob(pattern)


class Regex(Selector):
    def __init__(self, pattern):
        self.pattern = pattern

    def select(self, df):
        pat = re.compile(self.pattern)
        return [col for col in df.columns if pat.match(col) is not None]

    def __repr__(self):
        return f"regex({self.pattern!r})"


def regex(pattern):
    return Regex(pattern)


class Filter(Selector):
    def __init__(self, predicate):
        self.predicate = predicate

    def select(self, df):
        return [col for col in df.columns if self.predicate(df[col])]

    def __repr__(self):
        return f"filter({self.predicate!r})"


def filter(predicate):
    return Filter(predicate)


class FilterNames(Selector):
    def __init__(self, predicate):
        self.predicate = predicate

    def select(self, df):
        return [col for col in df.columns if self.predicate(col)]

    def __repr__(self):
        return f"filter_names({self.predicate!r})"


def filter_names(predicate):
    return FilterNames(predicate)


class Custom(Selector):
    def __init__(self, selector_function):
        self.selector_function = selector_function

    def select(self, df):
        return list(self.selector_function(df).columns)

    def __repr__(self):
        return f"custom(self.selector_function!r)"


def custom(selector_function):
    return Custom(selector_function)
