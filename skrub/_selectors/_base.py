def all():
    return All()


def cols(*columns):
    return Cols(columns)


def inv(obj):
    return all() - obj


def make_selector(obj):
    if isinstance(obj, Selector):
        return obj
    if isinstance(obj, str):
        return Cols([obj])
    if not hasattr(obj, "__iter__"):
        raise ValueError(f"selector not understood: {obj}")
    return Cols(obj)


def select(df, selector):
    return df.select(make_selector(selector).select(df))


class Selector:
    def select(self, df):
        raise NotImplementedError()

    def __invert__(self):
        return all() - self

    def __or__(self, other):
        return SetOp(self, other, "__or__")

    def __ror__(self, other):
        return self | other

    def __and__(self, other):
        return SetOp(self, other, "__and__")

    def __rand__(self, other):
        return self & other

    def __sub__(self, other):
        return SetOp(self, other, "__sub__")

    def __rsub__(self, other):
        return self - other


class All(Selector):
    def select(self, df):
        return list(df.columns)

    def __repr__(self):
        return "all()"


class Cols(Selector):
    def __init__(self, columns):
        self.columns = list(columns)

    def select(self, df):
        all_selected = set(self.columns)
        assert all_selected.issubset(df.columns)
        return [col for col in df.columns if col in all_selected]

    def __repr__(self):
        args = ", ".join(map(repr, self.columns))
        return f"cols({args})"


class SetOp(Selector):
    def __init__(self, left, right, op):
        self.left = make_selector(left)
        self.right = make_selector(right)
        self.op = op

    def select(self, df):
        left_selected = set(self.left.select(df))
        right_selected = set(self.right.select(df))
        all_selected = getattr(left_selected, self.op)(right_selected)
        return [col for col in df.columns if col in all_selected]

    def __repr__(self):
        op_repr = {
            "__or__": "|",
            "__and__": "&",
            "__sub__": "-",
        }[self.op]
        return f"({self.left!r} {op_repr} {self.right!r})"
