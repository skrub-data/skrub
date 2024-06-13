from .. import _dataframe as sbd
from .._dispatch import dispatch
from .._utils import repr_args


def all():
    """Select all columns.

    Examples
    --------
    >>> from skrub import _selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "height_mm": [297.0, 420.0],
    ...         "width_mm": [210.0, 297.0],
    ...         "kind": ["A4", "A3"],
    ...         "ID": [4, 3],
    ...     }
    ... )
    >>> df
       height_mm  width_mm kind  ID
    0      297.0     210.0   A4   4
    1      420.0     297.0   A3   3

    >>> s.select(df, s.all())
       height_mm  width_mm kind  ID
    0      297.0     210.0   A4   4
    1      420.0     297.0   A3   3

    """
    return All()


def cols(*columns):
    """Select columns by name.

    Examples
    --------
    >>> from skrub import _selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "height_mm": [297.0, 420.0],
    ...         "width_mm": [210.0, 297.0],
    ...         "kind": ["A4", "A3"],
    ...         "ID": [4, 3],
    ...     }
    ... )
    >>> df
       height_mm  width_mm kind  ID
    0      297.0     210.0   A4   4
    1      420.0     297.0   A3   3

    >>> s.select(df, s.cols('height_mm', 'ID'))
       height_mm  ID
    0      297.0   4
    1      420.0   3

    When this selector is used on its own, an error is raised if some columns
    are missing:

    >>> s.select(df, s.cols('width_mm', 'depth_mm'))
    Traceback (most recent call last):
        ...
    ValueError: The following columns are requested for selection but missing from dataframe: ['depth_mm']

    However, no error is raised when this selector is combined with other
    selectors:

    >>> s.select(df, s.all() & s.cols('width_mm', 'depth_mm'))
       width_mm
    0     210.0
    1     297.0

    In all skrub functions that accept a selector, a list of column names can
    be passed and ``cols`` will be used to turn it into a selector.

    >>> s.select(df, ['kind', 'ID'])
      kind  ID
    0   A4   4
    1   A3   3
    >>> s.make_selector(['kind', 'ID'])
    cols('kind', 'ID')
    >>> s.all() & ['kind', 'ID']
    (all() & cols('kind', 'ID'))
    """  # noqa: E501
    return Cols(columns)


def inv(obj):
    """Invert a selector.

    This selects all columns except those that are matched by the input; it is
    equivalent to ``all() - obj`` or ``~make_selector(obj)``. The argument
    ``obj`` can be a selector but also a column name or list of column names.

    Examples
    --------
    >>> from skrub import _selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "height_mm": [297.0, 420.0],
    ...         "width_mm": [210.0, 297.0],
    ...         "kind": ["A4", "A3"],
    ...         "ID": [4, 3],
    ...     }
    ... )
    >>> df
       height_mm  width_mm kind  ID
    0      297.0     210.0   A4   4
    1      420.0     297.0   A3   3

    >>> s.select(df, ['ID'])
       ID
    0   4
    1   3

    >>> s.select(df, s.inv(['ID']))
       height_mm  width_mm kind
    0      297.0     210.0   A4
    1      420.0     297.0   A3

    >>> s.select(df, ~s.cols('ID'))
       height_mm  width_mm kind
    0      297.0     210.0   A4
    1      420.0     297.0   A3

    """
    return ~make_selector(obj)


def make_selector(obj):
    """Transform a selector, column name or list of column names into a selector.

    Examples
    --------
    >>> from skrub import _selectors as s

    >>> s.make_selector('ID')
    cols('ID')

    >>> s.make_selector(['ID', 'kind'])
    cols('ID', 'kind')

    >>> s.make_selector(s.cols('ID', 'kind'))
    cols('ID', 'kind')

    """
    if isinstance(obj, Selector):
        return obj
    if isinstance(obj, str):
        return cols(obj)
    if not hasattr(obj, "__iter__"):
        raise ValueError(f"Selector not understood: {obj}")
    return cols(*obj)


@dispatch
def _select_col_names(df, col_names):
    raise NotImplementedError()


@_select_col_names.specialize("pandas", argument_type="DataFrame")
def _select_col_names_pandas(df, col_names):
    return df[col_names]


@_select_col_names.specialize("polars", argument_type="DataFrame")
def _select_col_names_polars(df, col_names):
    return df.select(col_names)


def select(df, selector):
    """Apply a selector to a dataframe and return the resulting dataframe.

    ``selector`` can be anything accepted by ``make_selector`` i.e. a selector,
    column name or list of column names.

    Examples
    --------
    >>> from skrub import _selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "height_mm": [297.0, 420.0],
    ...         "width_mm": [210.0, 297.0],
    ...         "kind": ["A4", "A3"],
    ...         "ID": [4, 3],
    ...     }
    ... )
    >>> df
       height_mm  width_mm kind  ID
    0      297.0     210.0   A4   4
    1      420.0     297.0   A3   3

    >>> selector = s.all() - 'ID'
    >>> selector
    (all() - cols('ID'))

    >>> selector.expand(df)
    ['height_mm', 'width_mm', 'kind']

    >>> s.select(df, selector)
       height_mm  width_mm kind
    0      297.0     210.0   A4
    1      420.0     297.0   A3

    We can also pass column names directly:

    >>> s.select(df, ['kind', 'ID'])
      kind  ID
    0   A4   4
    1   A3   3

    """
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
    def __init__(
        self, predicate, args=None, kwargs=None, name=None, selector_repr=None
    ):
        self.predicate = predicate
        self.args = () if args is None else args
        self.kwargs = {} if kwargs is None else kwargs
        self.name = name
        self.selector_repr = selector_repr

    def _matches(self, col):
        return self.predicate(col, *self.args, **self.kwargs)

    @staticmethod
    def _default_name():
        return "filter"

    def __repr__(self):
        if self.selector_repr is not None:
            return self.selector_repr
        if self.name is None:
            pred_name = getattr(self.predicate, "__qualname__", None) or repr(
                self.predicate
            )
            extra_args = repr_args(self.args, self.kwargs)
            args = ", ".join([pred_name, extra_args]) if extra_args else pred_name
            return f"{self._default_name()}({args})"
        return f"{self.name}({repr_args(self.args, self.kwargs)})"


def filter(predicate, *args, **kwargs):
    """Select columns for which ``predicate`` returns True.

    For each column ``col`` in the dataframe, ``predicate`` is called as
    ``predicate(col, *args, **kwargs)`` and the column is kept if it returns
    True. To filter columns based only on their name, see also
    ``filter_names``.

    ``args`` and ``kwargs`` are extra parameters for the predicate. Storing
    parameters like this rather than in a closure can help using an importable
    function as the predicate rather than a local one, which is necessary to
    pickle the selector. (An alternative is to use ``functools.partial``).

    Examples
    --------
    >>> from skrub import _selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "height_mm": [297.0, 420.0],
    ...         "width_mm": [210.0, 297.0],
    ...         "kind": ["A4", "A3"],
    ...         "ID": [4, 3],
    ...     }
    ... )
    >>> df
       height_mm  width_mm kind  ID
    0      297.0     210.0   A4   4
    1      420.0     297.0   A3   3

    >>> selector = s.filter(lambda col: 'A4' in col.values)
    >>> s.select(df, selector)
      kind
    0   A4
    1   A3

    >>> def contains(col, value):
    ...    return value in col.values

    >>> selector = s.filter(contains, 3)
    >>> selector
    filter(contains, 3)

    >>> s.select(df, selector)
       ID
    0   4
    1   3

    """
    return Filter(predicate, args=args, kwargs=kwargs)


class NameFilter(Filter):
    def _matches(self, col):
        return self.predicate(sbd.name(col), *self.args, **self.kwargs)

    @staticmethod
    def _default_name():
        return "filter_names"


def filter_names(predicate, *args, **kwargs):
    """Select columns based on their name.

    For a column whose name is ``col_name``, ``predicate`` is called as
    ``predicate(col_name, *args, **kwargs)`` and the column is selected if
    returns ``True``. Note this is different from ``filter``, because here the
    predicate is passed the column name whereas with ``filter``, the predicate
    is passed the actual column (pandas or polars Series).


    ``args`` and ``kwargs`` are extra parameters for the predicate. Storing
    parameters like this rather than in a closure can help using an importable
    function as the predicate rather than a local one, which is necessary to
    pickle the selector. (An alternative is to use ``functools.partial``).

    Examples
    --------
    >>> from skrub import _selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "height_mm": [297.0, 420.0],
    ...         "width_mm": [210.0, 297.0],
    ...         "kind": ["A4", "A3"],
    ...         "ID": [4, 3],
    ...     }
    ... )
    >>> df
       height_mm  width_mm kind  ID
    0      297.0     210.0   A4   4
    1      420.0     297.0   A3   3

    >>> selector = s.filter_names(lambda name: name.endswith('_mm'))
    >>> s.select(df, selector)
       height_mm  width_mm
    0      297.0     210.0
    1      420.0     297.0

    If we want to pickle the selector, we're better off using an importable
    function and passing the arguments separately:

    >>> selector = s.filter_names(str.endswith, '_mm')
    >>> selector
    filter_names(str.endswith, '_mm')

    >>> s.select(df, selector)
       height_mm  width_mm
    0      297.0     210.0
    1      420.0     297.0

    >>> import pickle
    >>> _ = pickle.dumps(selector) # OK

    """
    return NameFilter(predicate, args=args, kwargs=kwargs)
