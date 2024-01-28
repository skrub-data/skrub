from .. import _dataframe as sbd
from ._utils import list_difference, list_intersect


def all():
    return All()


def cols(*columns):
    return Cols(columns)


def inv(obj):
    return Inv(obj)


def make_selector(obj):
    if isinstance(obj, Selector):
        return obj
    if isinstance(obj, str):
        return Cols([obj])
    if not hasattr(obj, "__iter__"):
        raise ValueError(f"selector not understood: {obj}")
    return Cols(obj)


@sbd.dispatch
def _select_col_names(df, selector):
    raise NotImplementedError()


@_select_col_names.specialize("pandas")
def _select_col_names_pandas(df, col_names):
    return df[col_names]


@_select_col_names.specialize("polars")
def _select_col_names_polars(df, col_names):
    return df.select(col_names)


def select(df, selector):
    return _select_col_names(df, make_selector(selector).select(df))


class Selector:
    def select(self, df, ignore=()):
        raise NotImplementedError()

    def __invert__(self):
        return Inv(self)

    # TODO: implement short-circuit

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
    def select(self, df, ignore=()):
        return list_difference(sbd.column_names(df), ignore)

    def __repr__(self):
        return "all()"


class Cols(Selector):
    def __init__(self, columns):
        self.columns = list(columns)

    def select(self, df, ignore=()):
        all_selected = set(self.columns).difference(ignore)
        assert all_selected.issubset(sbd.column_names(df))
        return list_intersect(sbd.column_names(df), all_selected)

    def __repr__(self):
        args = ", ".join(map(repr, self.columns))
        return f"cols({args})"


class Inv(Selector):
    def __init__(self, complement):
        self.complement = make_selector(complement)

    def select(self, df, ignore=()):
        inv_selected = self.complement.select(df, ignore)
        return list_difference(sbd.column_names(df), set(inv_selected).union(ignore))

    def __repr__(self):
        return "~({self.complement!r})"


class Or(Selector):
    def __init__(self, left, right):
        self.left = make_selector(left)
        self.right = make_selector(right)

    def select(self, df, ignore=()):
        df_cols = sbd.column_names(df)
        left_selected = set(self.left.select(df, ignore))
        right_selected = self.right.select(df, ignore=left_selected.union(ignore))
        all_selected = left_selected.union(right_selected).difference(ignore)
        return list_intersect(df_cols, all_selected)

    def __repr__(self):
        return f"({self.left!r} | {self.right!r})"


class And(Selector):
    def __init__(self, left, right):
        self.left = make_selector(left)
        self.right = make_selector(right)

    def select(self, df, ignore=()):
        df_cols = sbd.column_names(df)
        left_selected = set(self.left.select(df, ignore))
        left_unselected = set(df_cols).difference(left_selected)
        right_selected = self.right.select(df, ignore=left_unselected.union(ignore))
        all_selected = left_selected.intersection(right_selected).difference(ignore)
        return list_intersect(df_cols, all_selected)

    def __repr__(self):
        return f"({self.left!r} & {self.right!r})"


class Sub(Selector):
    def __init__(self, left, right):
        self.left = make_selector(left)
        self.right = make_selector(right)

    def select(self, df, ignore=()):
        df_cols = sbd.column_names(df)
        left_selected = set(self.left.select(df, ignore))
        left_unselected = set(df_cols).difference(left_selected)
        right_selected = self.right.select(df, ignore=left_unselected.union(ignore))
        all_selected = left_selected.difference(right_selected).difference(ignore)
        return list_intersect(df_cols, all_selected)

    def __repr__(self):
        return f"({self.left!r} - {self.right!r})"


class XOr(Selector):
    def __init__(self, left, right):
        self.left = make_selector(left)
        self.right = make_selector(right)

    def select(self, df, ignore=()):
        df_cols = sbd.column_names(df)
        left_selected = self.left.select(df, ignore)
        right_selected = self.right.select(df, ignore)
        all_selected = set(left_selected).symmetric_difference(right_selected)
        return list_intersect(df_cols, all_selected.difference(ignore))

    def __repr__(self):
        return f"({self.left!r} ^ {self.right!r})"
