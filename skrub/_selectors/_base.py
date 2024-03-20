from .. import _dataframe as sbd
from .._dispatch import dispatch
from .._utils import repr_args


def all():
    return All()


def cols(*columns):
    return Cols(columns)


def inv(obj):
    return ~make_selector(obj)


def make_selector(obj):
    if isinstance(obj, Selector):
        return obj
    if isinstance(obj, str):
        return cols(obj)
    if not hasattr(obj, "__iter__"):
        raise ValueError(f"Selector not understood: {obj}")
    return cols(*obj)


@dispatch
def _select_col_names(df, selector):
    raise NotImplementedError()


@_select_col_names.specialize("pandas")
def _select_col_names_pandas(df, col_names):
    return df[col_names]


@_select_col_names.specialize("polars")
def _select_col_names_polars(df, col_names):
    return df.select(col_names)


def select(df, selector):
    return _select_col_names(df, make_selector(selector).expand(df))


class Selector:
    def _matches(self, col):
        raise NotImplementedError()

    def expand(self, df):
        matching_col_names = []
        for col_name in sbd.column_names(df):
            col = sbd.col(df, col_name)
            if self._matches(col):
                matching_col_names.append(col_name)
        return matching_col_names

    def __invert__(self):
        return Inv(self)

    def __or__(self, other):
        return Or(self, other)

    def __ror__(self, other):
        return Or(other, self)

    def __and__(self, other):
        return And(self, other)

    def __rand__(self, other):
        return And(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __xor__(self, other):
        return XOr(self, other)

    def __rxor__(self, other):
        return XOr(other, self)


class All(Selector):
    def _matches(self, col):
        return True

    def __repr__(self):
        return "all()"


def _check_string_list(columns):
    columns = list(columns)
    for c in columns:
        if not isinstance(c, str):
            raise ValueError(
                "Column name selector should be initialized with a list of str. Found"
                f" non-string element: {c!r}."
            )
    return columns


class Cols(Selector):
    def __init__(self, columns):
        self.columns = _check_string_list(columns)

    def _matches(self, col):
        return sbd.name(col) in self.columns

    def expand(self, df):
        missing = set(self.columns).difference(sbd.column_names(df))
        if missing:
            raise ValueError(
                "The following columns are requested for selection but "
                f"missing from dataframe: {list(missing)}"
            )
        return self.columns

    def __repr__(self):
        return f"cols({', '.join(map(repr, self.columns))})"


class Inv(Selector):
    def __init__(self, complement):
        self.complement = make_selector(complement)

    def _matches(self, col):
        return not self.complement._matches(col)

    def __repr__(self):
        return f"(~{self.complement!r})"


class Or(Selector):
    def __init__(self, left, right):
        self.left = make_selector(left)
        self.right = make_selector(right)

    def _matches(self, col):
        return self.left._matches(col) or self.right._matches(col)

    def __repr__(self):
        return f"({self.left!r} | {self.right!r})"


class And(Selector):
    def __init__(self, left, right):
        self.left = make_selector(left)
        self.right = make_selector(right)

    def _matches(self, col):
        return self.left._matches(col) and self.right._matches(col)

    def __repr__(self):
        return f"({self.left!r} & {self.right!r})"


class Sub(Selector):
    def __init__(self, left, right):
        self.left = make_selector(left)
        self.right = make_selector(right)

    def _matches(self, col):
        return self.left._matches(col) and (not self.right._matches(col))

    def __repr__(self):
        return f"({self.left!r} - {self.right!r})"


class XOr(Selector):
    def __init__(self, left, right):
        self.left = make_selector(left)
        self.right = make_selector(right)

    def _matches(self, col):
        return self.left._matches(col) ^ self.right._matches(col)

    def __repr__(self):
        return f"({self.left!r} ^ {self.right!r})"


class Filter(Selector):
    def __init__(self, predicate, args=None, kwargs=None, name=None):
        self.predicate = predicate
        self.args = () if args is None else args
        self.kwargs = {} if kwargs is None else kwargs
        self.name = name

    def _matches(self, col):
        return self.predicate(col, *self.args, **self.kwargs)

    @staticmethod
    def _default_name():
        return "filter"

    def __repr__(self):
        if self.name is None:
            args_r = repr_args((self.predicate,) + self.args, self.kwargs)
            return f"{self._default_name()}({args_r})"
        return f"{self.name}({repr_args(self.args, self.kwargs)})"


def filter(predicate, args=None, kwargs=None):
    return Filter(predicate, args=args, kwargs=kwargs)


class NameFilter(Filter):
    def _matches(self, col):
        return self.predicate(sbd.name(col), *self.args, **self.kwargs)

    @staticmethod
    def _default_name():
        return "filter_names"


def filter_names(predicate, args=None, kwargs=None):
    return NameFilter(predicate, args=args, kwargs=kwargs)
