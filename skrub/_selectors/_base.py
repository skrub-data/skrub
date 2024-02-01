from .. import _dataframe as sbd
from ._utils import list_difference, list_intersect


def all():
    return All()


def empty():
    return Empty()


def cols(*columns):
    if not columns:
        return empty()
    return Cols(columns)


def inv(obj):
    return ~make_selector(obj)


def make_selector(obj):
    if isinstance(obj, Selector):
        return obj
    if isinstance(obj, str):
        return cols(obj)
    if not hasattr(obj, "__iter__"):
        raise ValueError(f"selector not understood: {obj}")
    return cols(*obj)


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

    def __or__(self, other):
        return make_selector(other).__ror__(self)

    def __ror__(self, other):
        return Or(other, self)

    def __and__(self, other):
        return make_selector(other).__rand__(self)

    def __rand__(self, other):
        return And(other, self)

    def __sub__(self, other):
        return make_selector(other).__rsub__(self)

    def __rsub__(self, other):
        return Sub(other, self)

    def __xor__(self, other):
        return make_selector(other).__rxor__(self)

    def __rxor__(self, other):
        return XOr(other, self)


class All(Selector):
    def select(self, df, ignore=()):
        return list_difference(sbd.column_names(df), ignore)

    def __repr__(self):
        return "all()"

    def __invert__(self):
        return empty()

    def __or__(self, other):
        return all()

    def __ror__(self, other):
        return all()

    def __and__(self, other):
        return make_selector(other)

    def __rand__(self, other):
        return make_selector(other)

    def __sub__(self, other):
        return inv(other)

    def __rsub__(self, other):
        return empty()

    def __xor__(self, other):
        return inv(other)

    def __rxor__(self, other):
        return inv(other)


class Empty(Selector):
    def select(self, df, ignore=()):
        return []

    def __repr__(self):
        return "empty()"

    def __invert__(self):
        return all()

    def __or__(self, other):
        return make_selector(other)

    def __ror__(self, other):
        return make_selector(other)

    def __and__(self, other):
        return empty()

    def __rand__(self, other):
        return empty()

    def __sub__(self, other):
        return empty()

    def __rsub__(self, other):
        return make_selector(other)

    def __xor__(self, other):
        return make_selector(other)

    def __rxor__(self, other):
        return make_selector(other)


class Cols(Selector):
    def __init__(self, columns):
        columns = list(columns)
        for c in columns:
            if not isinstance(c, str):
                raise ValueError(
                    "Cols selector should be initialized with a list of str. Found"
                    f" non-string element: {c!r}."
                )
        self.columns = columns

    def select(self, df, ignore=()):
        all_selected = set(self.columns).difference(ignore)
        assert all_selected.issubset(sbd.column_names(df))
        return list_intersect(sbd.column_names(df), all_selected)

    def _set_op(self, other, op):
        other = make_selector(other)
        if not isinstance(other, Cols):
            return getattr(super(), op)(other)
        self_cols = set(self.columns)
        other_cols = set(other.columns)
        result_cols = getattr(self_cols, op)(other_cols)
        return cols(*sorted(result_cols))

    def __repr__(self):
        args = ", ".join(map(repr, self.columns))
        return f"cols({args})"

    def __or__(self, other):
        return self._set_op(other, "__or__")

    def __ror__(self, other):
        return self._set_op(other, "__ror__")

    def __and__(self, other):
        return self._set_op(other, "__and__")

    def __rand__(self, other):
        return self._set_op(other, "__rand__")

    def __sub__(self, other):
        return self._set_op(other, "__sub__")

    def __rsub__(self, other):
        return self._set_op(other, "__rsub__")

    def __xor__(self, other):
        return self._set_op(other, "__xor__")

    def __rxor__(self, other):
        return self._set_op(other, "__rxor__")


class Inv(Selector):
    def __init__(self, complement):
        self.complement = make_selector(complement)

    def select(self, df, ignore=()):
        inv_selected = self.complement.select(df, ignore)
        return list_difference(sbd.column_names(df), set(inv_selected).union(ignore))

    def __repr__(self):
        return f"~({self.complement!r})"


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
