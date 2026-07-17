"""
Defining new selectors
----------------------

This advanced section is aimed at skrub developers adding new selectors to
this module.

A Selector subclass must define the ``_matches`` method. It accepts a column and
returns True if the column should be selected.

Additionally, the subclass can override the ``expand`` method. It accepts a
dataframe and returns the list of column names that should be selected. This is
only called when the selector is used by itself. Whenever it is combined with
other selectors with operators, ``_matches`` is used. Overriding ``expand`` thus
allows special-casing the behavior when it is used on its own, such as raising
an exception when a simple list of column names is used for selection and some
are missing from the dataframe. Overriding ``expand`` is not necessary in most
cases; it may actually never be necessary except for the ``cols`` special case.

A simpler alternative to defining a new Selector subclass is to define a
function that constructs a selector by calling ``filter`` or ``filter_names`` with an
appropriate predicate and arguments; most selectors offered by this module are
implemented with this approach.

>>> from skrub import _dataframe as sbd
>>> from skrub import selectors as s
>>> import pandas as pd
>>> df = pd.DataFrame(
...     {
...         "height_mm": [297.0, 420.0],
...         "width_mm": [210.0, 297.0],
...         "kind": ["A4", "A3"],
...         "ID": [4, 3],
...     }
... )

Defining a new class:

>>> class EndsWith(s.Selector):
...     def __init__(self, suffix):
...         self.suffix = suffix
...
...     def _matches(self, col):
...         return sbd.name(col).endswith(self.suffix)

>>> EndsWith('_mm').expand(df)
['height_mm', 'width_mm']

Using a filter:

>>> def ends_with(suffix):
...     return s.filter_names(str.endswith, suffix)

>>> ends_with('_mm').expand(df)
['height_mm', 'width_mm']

>>> ends_with('_mm')
filter_names(str.endswith, '_mm')

Directly instantiating a Filter or FilterNames object allows passing the name
argument and thus controlling the repr of the resulting selector, so an
slightly improved version could be:

>>> from skrub.selectors._base import NameFilter

>>> def ends_with(suffix):
...     return NameFilter(str.endswith, args=(suffix,), name='ends_with')

>>> ends_with('_mm')
ends_with('_mm')

>>> ends_with('_mm').expand(df)
['height_mm', 'width_mm']
"""

from .. import _dataframe as sbd
from .._dispatch import dispatch, raise_dispatch_unregistered_type
from .._utils import repr_args


def all():
    """Select all columns in a dataframe.

    See Also
    --------
    inv : Invert a selector
    cols : Select columns by exact name

    Examples
    --------
    >>> from skrub import selectors as s
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

    Select all columns:

    >>> s.select(df, s.all())
       height_mm  width_mm kind  ID
    0      297.0     210.0   A4   4
    1      420.0     297.0   A3   3

    Use ``all()`` as a base for excluding columns, for example by name:

    >>> s.select(df, s.all() - 'ID')
       height_mm  width_mm kind
    0      297.0     210.0   A4
    1      420.0     297.0   A3
    """
    return All()


def cols(*columns):
    """Select columns by name.

    The selected columns are returned in the order they are listed in ``columns``,
    not the order they appear in the dataframe. If any of the requested columns are
    missing from the dataframe, an exception is raised.

    See Also
    --------
    all : Select all columns
    glob : Select columns by UNIX-like name pattern
    regex : Select columns by regular expression pattern

    Examples
    --------
    >>> from skrub import selectors as s
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
    ValueError: The following columns are requested for selection but missing
    from dataframe: ['depth_mm']

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

    Returns
    -------
    Selector
        A ``Selector`` that is the inverse of the input selector.

    See Also
    --------
    all : Select all columns

    Notes
    -----
    ``inv`` is primarily a convenience function. When working with a single
    selector or column name, ``~selector`` may be more readable. When excluding
    from the full set of columns, ``all() - selector`` is more explicit.

    Examples
    --------
    >>> from skrub import selectors as s
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
    """Normalize a selector, column name, or list of names into a ``Selector``\
    object.

    This function converts user input into a consistent selector object:

    - Selectors are returned as-is
    - Strings are converted to ``cols(name)``
    - Lists of strings are converted to ``cols(*names)``

    Parameters
    ----------
    obj : selector, str, or list
        The object to normalize.

    Returns
    -------
    Selector
        A ``Selector`` object (or subclass).

    See Also
    --------
    cols : Select specific columns by name

    Examples
    --------

    This function is used to normalize user input so that the result is a
    selector compatible with the rest of the API.

    >>> from skrub import selectors as s

    Normalize a column name or list of column names:

    >>> s.make_selector('ID')
    cols('ID')
    >>> s.make_selector(['ID', 'kind'])
    cols('ID', 'kind')

    A selector is returned unchanged:

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
    raise_dispatch_unregistered_type(df, kind="DataFrame")


@_select_col_names.specialize("pandas", argument_type="DataFrame")
def _select_col_names_pandas(df, col_names):
    return df[col_names]


@_select_col_names.specialize("polars", argument_type="DataFrame")
def _select_col_names_polars(df, col_names):
    return df.select(col_names)


def select(df, selector):
    """Select the columns of a dataframe that are matched by the selector.

    This function returns a new dataframe containing only the columns of `df`
    matched by `selector`.

    Parameters
    ----------
    df : dataframe
        The dataframe to select columns from (pandas or polars).
    selector : selector, str, or list
        A selector object, single column name, or list of column names.

    Returns
    -------
    dataframe
        A new dataframe containing only the columns matched by the selector.

    See Also
    --------
    drop : Return all columns except those matched by a selector
    Selector.expand : Get the column names matched by a selector as a list

    Notes
    -----
    ``select`` is a convenience function that combines two operations:

    1. ``selector.expand(df)`` - Get list of matching column names
    2. Return the dataframe subset to those columns

    If you only need the list of matching column names (without subsetting the
    dataframe), use ``selector.expand(df)`` directly.

    Examples
    --------
    >>> from skrub import selectors as s
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

    Select all columns except 'ID':

    >>> selector = s.all() - 'ID'
    >>> selector
    (all() - cols('ID'))

    >>> s.select(df, selector)
       height_mm  width_mm kind
    0      297.0     210.0   A4
    1      420.0     297.0   A3

    Pass column names directly:

    >>> s.select(df, ['kind', 'ID'])
      kind  ID
    0   A4   4
    1   A3   3

    Select by dtype:

    >>> s.select(df, s.numeric())
       height_mm  width_mm  ID
    0      297.0     210.0   4
    1      420.0     297.0   3

    Combine multiple selectors:

    >>> s.select(df, s.numeric() & s.glob('*_mm'))
       height_mm  width_mm
    0      297.0     210.0
    1      420.0     297.0

    """
    return _select_col_names(df, make_selector(selector).expand(df))


def drop(df, selector):
    """Select the columns of a dataframe that are NOT matched by the selector.

    This is the complement of ``select()``: it returns a new dataframe containing
    all columns except those matched by the selector.

    Parameters
    ----------
    df : dataframe
        The dataframe to process.
    selector : selector, str, or list
        A selector object, single column name, or list of column names indicating
        which columns to drop.

    Returns
    -------
    dataframe
        A new dataframe with the matched columns removed, preserving the order
        of remaining columns.

    See Also
    --------
    select : Return only the columns matched by a selector
    inv : Create an inverted selector matching all columns except those from the input

    Notes
    -----
    ``drop`` is logically equivalent to ``select(df, ~selector)`` or
    ``select(df, s.all() - selector)``.

    ``drop`` preserves the original column order of the remaining columns.

    Examples
    --------
    >>> from skrub import selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "height_mm": [210.0, 297.0],
    ...         "width_mm": [188.5, 210.0],
    ...         "kind": ["A5", "A4"],
    ...         "ID": [5, 4],
    ...     }
    ... )
    >>> df
       height_mm  width_mm kind  ID
    0      210.0     188.5   A5   5
    1      297.0     210.0   A4   4

    Drop columns matching a pattern:

    >>> s.drop(df, s.glob("*_mm"))
      kind  ID
    0   A5   5
    1   A4   4

    Drop specific columns by name (can pass names directly):

    >>> s.drop(df, ['height_mm', 'width_mm'])
      kind  ID
    0   A5   5
    1   A4   4

    Preserve only certain types (via drop):

    >>> s.drop(df, s.all() - s.string())
      kind
    0   A5
    1   A4
    """
    all_cols = sbd.column_names(df)
    matched_cols = make_selector(selector).expand(df)
    remaining_cols = [col for col in all_cols if col not in matched_cols]
    return _select_col_names(df, remaining_cols)


class Selector:
    """Pattern for matching columns in a dataframe.

    A ``Selector`` is a reusable rule for selecting columns based on various
    criteria (data type, name pattern, content properties, etc.). Selectors
    enable delayed selection: you can define a selection rule before the data
    is available.

    **How to use selectors**

    - **Direct selection:** ``s.select(df, selector)`` returns a filtered dataframe
    - **With** :class:`skrub.ApplyToCols`: ``ApplyToCols(transformer, cols=selector)``
      applies a transformer to selected columns
    - **In DataOps:** ``skrub.X(df).skb.apply(transformer, cols=selector)``
    - **Manual expansion:** ``selector.expand(df)`` gets column names for manual use

    **Combining Selectors**

    Selectors can be combined with set operators to create complex selection rules:

    - ``s.numeric() | s.boolean()`` - numeric OR boolean columns
    - ``s.all() - s.glob('*_id')`` - all columns except those whose name ends with "_id"
    - ``~s.cardinality_below(10)`` - high-cardinality columns

    .. note::

        This class is not meant to be instantiated manually. Create selectors using
        builder functions such as :func:`skrub.selectors.all()`,
        :func:`skrub.selectors.cols()`, :func:`skrub.selectors.glob()`, etc.

    See Also
    --------
    skrub.ApplyToCols :
        Apply a transformer only to some columns, possibly selected by a selector.
    skrub.DataOp.skb.apply :
        Apply a transformer to selected columns in a DataOps workflow.

    Examples
    --------
    >>> from skrub import selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'height_mm': [297.0, 420.0],
    ...     'width_mm': [210.0, 297.0],
    ...     'kind': ['A4', 'A3'],
    ...     'ID': [4, 3],
    ... })

    Use a selector to get the names of matching columns:

    >>> s.numeric().expand(df)
    ['height_mm', 'width_mm', 'ID']

    Selectors can be combined to create complex selection rules. For example,
    to select numeric columns but exclude the 'ID' column:

    >>> (s.numeric() - 'ID').expand(df)
    ['height_mm', 'width_mm']

    Use in data transformations, for example to apply a scaler only to numeric columns:

    >>> from skrub import ApplyToCols
    >>> from sklearn.preprocessing import StandardScaler
    >>> ApplyToCols(StandardScaler(), cols=s.numeric()).fit_transform(df)
    kind  height_mm  width_mm   ID
    0   A4       -1.0      -1.0  1.0
    1   A3        1.0       1.0 -1.0
    """

    def _matches(self, col):
        """Check if a column should be selected.

        Subclasses must override this method to define their selection criteria.

        Parameters
        ----------
        col : column object
            A column from the dataframe (pandas Series, polars Series, etc.).

        Returns
        -------
        bool
            ``True`` if the column should be selected, ``False`` otherwise.
        """
        raise NotImplementedError()

    def expand(self, df):
        """Get the names of columns matched by the selector in a list.

        This method evaluates the selector's matching criteria against each column
        in the dataframe and returns the names of columns that match. This can be
        useful to extract column names to be used in dataframe operations.

        Parameters
        ----------
        df : dataframe
            A pandas or polars dataframe to evaluate the selector against.

        Returns
        -------
        list of str
            The names of columns from ``df`` that the selector matches.

        See Also
        --------
        expand_index : Get indices of matching columns instead of names.

        Examples
        --------
        >>> import pandas as pd
        >>> from skrub import selectors as s
        >>> some_selector = ~s.glob("*_mm")
        >>> df = pd.DataFrame(
        ...     {
        ...         "height_mm": [210.0, 297.0],
        ...         "width_mm": [188.5, 210.0],
        ...         "kind": ["A5", "A4"],
        ...         "ID": [5, 4],
        ...     }
        ... )
        >>> some_selector.expand(df)
        ['kind', 'ID']

        Use to select columns in a dataframe:

        >>> df[some_selector.expand(df)]
           kind  ID
        0   A5   5
        1   A4   4

        """
        matching_col_names = []
        for col_name in sbd.column_names(df):
            col = sbd.col(df, col_name)
            if self._matches(col):
                matching_col_names.append(col_name)
        return matching_col_names

    def expand_index(self, df):
        """Get the indices of columns matched by the selector in a list.

        This method evaluates the selector against each column and returns the
        positional indices (0, 1, 2, ...) of matching columns instead of their names.

        Parameters
        ----------
        df : dataframe
            A pandas or polars dataframe to evaluate the selector against.

        Returns
        -------
        list of int
            The indices (zero-indexed) of columns from ``df`` that the
            selector matches.

        See Also
        --------
        expand : Get names of matching columns instead of indices.

        Examples
        --------
        >>> import pandas as pd
        >>> from skrub import selectors as s
        >>> some_selector = ~s.glob("*_mm")
        >>> df = pd.DataFrame(
        ...     {
        ...         "height_mm": [210.0, 297.0],
        ...         "width_mm": [188.5, 210.0],
        ...         "kind": ["A5", "A4"],
        ...         "ID": [5, 4],
        ...     }
        ... )
        >>> some_selector.expand_index(df)
        [2, 3]

        """
        matching_col_indices = []
        for col_idx, col in enumerate(sbd.to_column_list(df)):
            if self._matches(col):
                matching_col_indices.append(col_idx)
        return matching_col_indices

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
    """Select columns for which a custom predicate function returns ``True``.

    This allows selecting columns based on an arbitrary criterion. The predicate takes
    as input the column (a pandas or polars Series) and must return True if it
    should be selected and False otherwise.

    For example, this selector can be used when built-in selectors don't match
    your selection criterion, to select columns based on their content or computed
    properties (e.g., variance, range...), or to combine multiple criteria in a
    single predicate.

    Parameters
    ----------
    predicate : callable
        A function that takes a column and optional extra arguments, returning
        ``True`` to select the column or ``False`` to exclude it.
        Signature: ``predicate(col, *args, **kwargs) -> bool``
    *args : tuple
        Extra positional arguments passed to the predicate.
    **kwargs : dict
        Extra keyword arguments passed to the predicate.

    Returns
    -------
    Selector
        A ``Filter`` selector that matches columns where ``predicate`` returns ``True``.

    See Also
    --------
    filter_names : Select columns based only on their name (not content)

    Notes
    -----
    For a column ``col``, the predicate is called as::

        predicate(col, *args, **kwargs)

    The column is kept if ``predicate`` returns ``True``.

    To pickle the selector, use importable functions as predicates
    rather than lambdas. Prefer passing parameters via ``*args`` or ``**kwargs``::

        # Picklable (importable function + explicit args)
        s.filter(str.startswith, 'prefix')

        # Picklable alternative (with functools.partial)
        import functools
        s.filter(functools.partial(str.startswith, 'prefix'))

        # NOT picklable (closure)
        prefix = 'test'
        s.filter(lambda col: str(col.name).startswith(prefix))

    For name-based selection, consider using :func:`filter_names`, :func:`glob`,
    or :func:`regex` instead of ``filter`` for simpler selection.

    Examples
    --------
    >>> from skrub import selectors as s
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

    Select columns containing a specific value:

    >>> selector = s.filter(lambda col: 'A4' in col.values)
    >>> s.select(df, selector)
      kind
    0   A4
    1   A3

    Use an importable function with explicit arguments (picklable):

    >>> def contains(col, value):
    ...    return value in col.values

    >>> selector = s.filter(contains, 3)
    >>> selector
    filter(contains, 3)

    >>> s.select(df, selector)
       ID
    0   4
    1   3

    Combine with type selectors:

    >>> def has_high_range(col):
    ...     return col.max() - col.min() > 100

    >>> s.select(df, s.numeric() & s.filter(has_high_range))
       height_mm
    0      297.0
    1      420.0

    Note that ``s.numeric()`` short-circuits the evaluation of ``has_high_range``
    for non-numeric columns, so the predicate is only called for numeric columns,
    avoiding errors on string columns.

    """
    return Filter(predicate, args=args, kwargs=kwargs)


class NameFilter(Filter):
    def _matches(self, col):
        return self.predicate(sbd.name(col), *self.args, **self.kwargs)

    @staticmethod
    def _default_name():
        return "filter_names"


def filter_names(predicate, *args, **kwargs):
    r"""Select columns based only on their name.

    The predicate takes as input the column name and must return True if it
    should be selected and False otherwise.

    Parameters
    ----------
    predicate : callable
        A function that takes a column name (string) and optional extra arguments,
        returning ``True`` to select the column or ``False`` to exclude it.
        Signature: ``predicate(col_name, *args, **kwargs) -> bool``
    *args : tuple
        Extra positional arguments passed to the predicate. Using explicit
        arguments (instead of closures) helps with pickling the selector.
    **kwargs : dict
        Extra keyword arguments passed to the predicate. Using explicit
        arguments (instead of closures) helps with pickling the selector.

    Returns
    -------
    Selector
        A ``NameFilter`` selector that matches columns where
        ``predicate(name, *args, **kwargs)`` returns ``True``.

    See Also
    --------
    filter : Select columns based on their content or properties
    glob : Select columns by wildcard pattern matching
    regex : Select columns by regular expression pattern matching

    Notes
    -----
    ``filter_names`` passes the column NAME
    (a string), while ``filter`` passes the column OBJECT (Series/DataFrame)

    For common patterns, prefer simpler selectors:

    - ``s.glob('*_mm')`` instead of ``s.filter_names(lambda n: n.endswith('_mm'))``
    - ``s.regex(r'col_\d+')`` instead of ``s.filter_names(lambda n: re.match(...))``

    To pickle the selector, use importable functions as predicates
    rather than lambdas. Pass parameters via ``*args`` or ``**kwargs``::

        # Picklable (importable function + explicit args)
        s.filter_names(str.endswith, '_mm')

        # Picklable alternative (with functools.partial)
        import functools
        s.filter_names(functools.partial(str.endswith, '_mm'))

        # NOT picklable (closure)
        suffix = '_mm'
        s.filter_names(lambda name: name.endswith(suffix))

    Examples
    --------
    >>> from skrub import selectors as s
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

    Prefer using *args and **kwargs to pass extra arguments to the predicate,
    rather than defining a dynamic function which may cause pickling errors.

    Select columns ending with a suffix (using lambda):

    >>> selector = s.filter_names(lambda name: name.endswith('_mm'))
    >>> s.select(df, selector)
       height_mm  width_mm
    0      297.0     210.0
    1      420.0     297.0

    Use an importable function with explicit arguments (picklable):

    >>> selector = s.filter_names(str.endswith, '_mm')
    >>> selector
    filter_names(str.endswith, '_mm')

    >>> s.select(df, selector)
       height_mm  width_mm
    0      297.0     210.0
    1      420.0     297.0

    >>> import pickle
    >>> _ = pickle.dumps(selector)  # Pickling works!

    Select columns with uppercase names:

    >>> s.select(df, s.filter_names(str.isupper))
       ID
    0   4
    1   3

    Combine with type selectors:

    >>> s.select(df, s.numeric() & s.filter_names(str.islower))
       height_mm  width_mm
    0      297.0     210.0
    1      420.0     297.0

    """
    return NameFilter(predicate, args=args, kwargs=kwargs)
