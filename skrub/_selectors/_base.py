from .. import _dataframe as sbd
from ._utils import list_difference, list_intersect


def all():
    return All()


def nothing():
    return Nothing()


def cols(*columns):
    if not columns:
        return nothing()
    return ExactCols(columns)


def name_in(*columns):
    if not columns:
        return nothing()
    return NameIn(columns)


def inv(obj):
    return ~_make_selector_in_expr(obj)


def make_selector(obj):
    if isinstance(obj, Selector):
        return obj
    if isinstance(obj, str):
        return cols(obj)
    if not hasattr(obj, "__iter__"):
        raise ValueError(f"selector not understood: {obj}")
    return cols(*obj)


def _make_selector_in_expr(obj):
    return make_selector(obj)._in_expr()


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

    def _in_expr(self):
        return self

    def __invert__(self):
        return Inv(self)

    def __or__(self, other):
        return _make_selector_in_expr(other).__ror__(self)

    def __ror__(self, other):
        if not isinstance(other, Selector):
            return _make_selector_in_expr(other) | self
        return Or(other, self)

    def __and__(self, other):
        return _make_selector_in_expr(other).__rand__(self)

    def __rand__(self, other):
        if not isinstance(other, Selector):
            return _make_selector_in_expr(other) & self
        return And(other, self)

    def __sub__(self, other):
        return _make_selector_in_expr(other).__rsub__(self)

    def __rsub__(self, other):
        if not isinstance(other, Selector):
            return _make_selector_in_expr(other) - self
        return Sub(other, self)

    def __xor__(self, other):
        return _make_selector_in_expr(other).__rxor__(self)

    def __rxor__(self, other):
        if not isinstance(other, Selector):
            return _make_selector_in_expr(other) ^ self
        return XOr(other, self)

    def use(self, transformer, n_jobs=None, columnwise="auto"):
        from .._on_column_selection import OnColumnSelection
        from .._on_each_column import OnEachColumn

        if isinstance(columnwise, str) and columnwise == "auto":
            columnwise = hasattr(transformer, "__single_column_transformer__")

        if columnwise:
            return OnEachColumn(transformer, cols=self, n_jobs=n_jobs)
        return OnColumnSelection(transformer, cols=self)


class All(Selector):
    def select(self, df, ignore=()):
        return list_difference(sbd.column_names(df), ignore)

    def __repr__(self):
        return "all()"

    def __invert__(self):
        return nothing()

    def __or__(self, other):
        return all()

    def __ror__(self, other):
        return all()

    def __and__(self, other):
        return _make_selector_in_expr(other)

    def __rand__(self, other):
        return _make_selector_in_expr(other)

    def __sub__(self, other):
        return inv(other)

    def __rsub__(self, other):
        return nothing()

    def __xor__(self, other):
        return inv(other)

    def __rxor__(self, other):
        return inv(other)


class Nothing(Selector):
    def select(self, df, ignore=()):
        return []

    def __repr__(self):
        return "()"

    def __invert__(self):
        return all()

    def __or__(self, other):
        return _make_selector_in_expr(other)

    def __ror__(self, other):
        return _make_selector_in_expr(other)

    def __and__(self, other):
        return nothing()

    def __rand__(self, other):
        return nothing()

    def __sub__(self, other):
        return nothing()

    def __rsub__(self, other):
        return _make_selector_in_expr(other)

    def __xor__(self, other):
        return _make_selector_in_expr(other)

    def __rxor__(self, other):
        return _make_selector_in_expr(other)


def _check_string_list(columns):
    columns = list(columns)
    for c in columns:
        if not isinstance(c, str):
            raise ValueError(
                "Column name selector should be initialized with a list of str. Found"
                f" non-string element: {c!r}."
            )
    return columns


class NameIn(Selector):
    def __init__(self, columns):
        self.columns = _check_string_list(columns)

    def select(self, df, ignore=()):
        all_selected = set(self.columns).difference(ignore)
        return list_intersect(sbd.column_names(df), all_selected)

    def _set_op(self, other, op):
        other = _make_selector_in_expr(other)
        if not isinstance(other, NameIn):
            return getattr(super(), op)(other)
        self_cols = set(self.columns)
        other_cols = set(other.columns)
        result_cols = getattr(self_cols, op)(other_cols)
        return name_in(*sorted(result_cols))

    def __repr__(self):
        args = ", ".join(map(repr, self.columns))
        return f"name_in({args})"

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


class ExactCols(Selector):
    def __init__(self, columns):
        self.columns = _check_string_list(columns)

    def select(self, df, ignore=()):
        all_selected = set(self.columns).difference(ignore)
        missing = all_selected.difference(sbd.column_names(df))
        if missing:
            raise ValueError(
                "The following columns are requested for selection but "
                f"missing from dataframe: {list(missing)}"
            )
        return list_intersect(sbd.column_names(df), all_selected)

    def _in_expr(self):
        return NameIn(self.columns)

    def __repr__(self):
        return repr(self.columns)


class Inv(Selector):
    def __init__(self, complement):
        self.complement = _make_selector_in_expr(complement)

    def select(self, df, ignore=()):
        inv_selected = self.complement.select(df, ignore)
        return list_difference(sbd.column_names(df), set(inv_selected).union(ignore))

    def __repr__(self):
        return f"~({self.complement!r})"


class Or(Selector):
    def __init__(self, left, right):
        self.left = _make_selector_in_expr(left)
        self.right = _make_selector_in_expr(right)

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
        self.left = _make_selector_in_expr(left)
        self.right = _make_selector_in_expr(right)

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
        self.left = _make_selector_in_expr(left)
        self.right = _make_selector_in_expr(right)

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
        self.left = _make_selector_in_expr(left)
        self.right = _make_selector_in_expr(right)

    def select(self, df, ignore=()):
        df_cols = sbd.column_names(df)
        left_selected = self.left.select(df, ignore)
        right_selected = self.right.select(df, ignore)
        all_selected = set(left_selected).symmetric_difference(right_selected)
        return list_intersect(df_cols, all_selected.difference(ignore))

    def __repr__(self):
        return f"({self.left!r} ^ {self.right!r})"
