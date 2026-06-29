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

    This is the most general selector, matching every column regardless of name,
    type, or content. It is useful as a starting point for building complex
    selections via operators.

    **When to use**

    - As a starting point for complex selections: ``s.all() & s.numeric()``
    - To exclude specific columns: ``s.all() - ['ID', 'index']``
    - To explicitly select everything (rarely needed, but more readable)
    - As the complement of other selectors via ``~s.some_selector()``

    Description
    -----------
    Returns a selector that matches all columns in the dataframe. This is useful
    for operations that need a baseline for composition with other selectors,
    such as selecting all columns except those matching a pattern.

    See Also
    --------
    inv : Select all columns except those matched by a selector
    cols : Select columns by explicit name
    make_selector : Convert a selector, name, or list to a selector

    Notes
    -----

    Operator combinations
    ~~~~~~~~~~~~~~~~~~~~~
    - ``s.all() & s.numeric()`` → All numeric columns
    - ``s.all() - 'ID'`` → All except 'ID'
    - ``s.all() & s.glob('*_mm')`` → All columns matching pattern

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

    Use ``all()`` as a base for exclusion:

    >>> s.select(df, s.all() - 'ID')
       height_mm  width_mm kind
    0      297.0     210.0   A4
    1      420.0     297.0   A3

    Combine with type selectors:

    >>> s.select(df, s.all() & s.numeric())
       height_mm  width_mm  ID
    0      297.0     210.0   4
    1      420.0     297.0   3
    """
    return All()


def cols(*columns):
    """Select columns by name.

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
    """Normalize a selector, column name, or list of names into a ``Selector``
    object.

    This function serves as the gateway function for all selector-accepting
    APIs in skrub. It is used internally by skrub to normalize user input into a
    consistent selector object:

    - Selectors are returned as-is
    - Strings are converted to ``cols(name)``
    - Lists of strings are converted to ``cols(*names)``

    Parameters
    ----------
    obj : selector, str, or list
        The object to normalize:
        - A ``Selector`` object: returned as-is
        - A string: converted to ``cols(name)``
        - A list/iterable of strings: converted to ``cols(*names)``

    Returns
    -------
    Selector
        A ``Selector`` object (or subclass).

    Raises
    ------
    ValueError
        If ``obj`` is not a selector, string, or iterable.

    See Also
    --------
    cols : Select specific columns by name
    select : Apply a selector and return matching columns
    drop : Apply a selector and return non-matching columns

    Examples
    --------
    >>> from skrub import selectors as s

    Normalize a single column name:

    >>> s.make_selector('ID')
    cols('ID')

    Normalize a list of column names:

    >>> s.make_selector(['ID', 'kind'])
    cols('ID', 'kind')

    A selector is returned unchanged:

    >>> s.make_selector(s.cols('ID', 'kind'))
    cols('ID', 'kind')

    Use with operators:

    >>> s.make_selector('ID') | s.numeric()
    (cols('ID') | numeric())

    This is useful in APIs that accept flexible column specifications:

    >>> from skrub import selectors as s
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

    These all work interchangeably:

    >>> s.select(df, s.cols('a'))
       a
    0  1
    1  2

    >>> s.select(df, 'a')  # Converted via make_selector
       a
    0  1
    1  2

    >>> s.select(df, ['a'])  # Also converted via make_selector
       a
    0  1
    1  2
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
    """Apply a selector to a dataframe and return the selected columns.

    This is the primary function for materializing selector expressions into
    actual column subsets. It evaluates a selector against a specific dataframe
    and returns a new dataframe containing only the columns matched by the selector.

    **When to use**

    - To extract columns matching a pattern or criterion from a dataframe
    - To prepare data for modeling (e.g., select features for a pipeline)
    - To inspect which columns a selector expression matches
    - Combined with selectors in exploratory data analysis workflows

    Parameters
    ----------
    df : dataframe
        The dataframe to select columns from (pandas or polars).
    selector : selector, str, or list
        A selector object, single column name, or list of column names.
        Passed through ``make_selector`` for normalization.

    Returns
    -------
    dataframe
        A new dataframe containing only the columns matched by the selector.
        The column order matches the selector's expansion order.

    See Also
    --------
    drop : Return all columns except those matched by a selector
    make_selector : Normalize column specifications into a selector
    Selector.expand : Get the column names matched by a selector (without
    subsetting)

    Notes
    -----
    ``select`` is a convenience function that combines two operations:

    1. ``selector.expand(df)`` - Get list of matching column names
    2. Return the dataframe subset to those columns

    If you only need the list of matching column names (without subsetting the
    dataframe), use ``selector.expand(df)`` directly.

    Operator combinations
    ~~~~~~~~~~~~~~~~~~~~~
    Complex selections use operators before passing to ``select``:

    - ``s.select(df, s.numeric() & s.glob('*_mm'))`` → Numeric columns with _mm
      suffix
    - ``s.select(df, s.all() - s.glob('ID'))`` → All except ID columns
    - ``s.select(df, s.numeric() | s.string())`` → Numeric or string columns
    - ``s.select(df, ~s.has_nulls())`` → Columns without missing values

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

    Build and apply a complex selector:

    >>> selector = s.all() - 'ID'
    >>> selector
    (all() - cols('ID'))

    >>> s.select(df, selector)
       height_mm  width_mm kind
    0      297.0     210.0   A4
    1      420.0     297.0   A3

    Pass column names directly (implicitly converted):

    >>> s.select(df, ['kind', 'ID'])
      kind  ID
    0   A4   4
    1   A3   3

    Select by type:

    >>> s.select(df, s.numeric())
       height_mm  width_mm  ID
    0      297.0     210.0   4
    1      420.0     297.0   3

    Combine multiple criteria:

    >>> s.select(df, s.all() & s.glob('*_mm'))
       height_mm  width_mm
    0      297.0     210.0
    1      420.0     297.0

    Use to explore selector matches:

    >>> s.select(df, s.string())
      kind
    0   A4
    1   A3
    """
    return _select_col_names(df, make_selector(selector).expand(df))


def drop(df, selector):
    """Apply a selector to a dataframe and return it without the selected columns.

    This is the complement of ``select()``: it returns a new dataframe containing
    all columns except those matched by the selector. Useful for removing unwanted
    columns (identifiers, metadata, noise) before modeling or analysis.

    **When to use**
    -----------
    - Remove ID/index columns before fitting models
    - Exclude metadata or internal columns from analysis
    - Drop all columns of a certain type (e.g., all nulls)
    - Clean data by removing problematic columns

    Parameters
    ----------
    df : dataframe
        The dataframe to process (pandas or polars).
    selector : selector, str, or list
        A selector object, single column name, or list of column names indicating
        which columns to drop. Passed through ``make_selector`` for normalization.

    Returns
    -------
    dataframe
        A new dataframe with the matched columns removed, preserving the order
        of remaining columns.

    See Also
    --------
    select : Return only the columns matched by a selector
    make_selector : Normalize column specifications into a selector
    inv : Create an inverted selector matching all columns except those from the input

    Notes
    -----
    ``drop`` is logically equivalent to ``select(df, ~selector)`` or
    ``select(df, s.all() - selector)``. The difference is that ``drop`` is more
    readable when your intent is to remove columns rather than specify what to keep.

    ``drop`` preserves the original column order of the remaining columns.

    Operator combinations
    ~~~~~~~~~~~~~~~~~~~~~
    Use operators to drop complex column selections:

    - ``s.drop(df, s.glob('ID*'))`` → Remove all ID columns
    - ``s.drop(df, s.numeric() & s.has_nulls())`` → Remove numeric cols with nulls
    - ``s.drop(df, s.all() - s.string())`` → Keep only string columns
    - ``s.drop(df, s.filter(lambda col: len(col) < 10))`` → Drop short columns

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

    Drop by type:

    >>> s.drop(df, s.numeric())
      kind
    0   A5
    1   A4

    Drop columns matching a predicate:

    >>> s.drop(df, s.filter_names(str.startswith, 'h'))
       width_mm kind  ID
    0      188.5   A5   5
    1      210.0   A4   4

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
    """Generic selector type that selects columns from a dataframe.

    A ``Selector`` is a reusable rule for selecting columns based on various
    criteria (data type, name pattern, content properties, etc.). Selectors
    enable delayed selection: you can define a selection rule before the data
    is available, making them ideal for scikit-learn pipelines and data
    processing workflows.

    This class is not meant to be instantiated manually. Create selectors using
    builder functions such as :func:`skrub.selectors.all()`,
    :func:`skrub.selectors.numeric()`, :func:`skrub.selectors.glob()`, etc.

    **How Selectors Work**

    For each column in a dataframe, a selector evaluates the ``_matches(column)``
    method:

    - Returns ``True`` → column is selected
    - Returns ``False`` → column is excluded

    **Key Methods**

    - :meth:`expand` - Get list of column names that the selector would select
    - :meth:`expand_index` - Get indices of columns that the selector would select
    - Operator support: ``|`` (OR), ``&`` (AND), ``-`` (except), ``^``
      (XOR), ``~`` (NOT)

    **Ways to Use Selectors**

    1. **Direct selection:** ``s.select(df, selector)`` returns a filtered dataframe
    2. **In transformers:** ``ApplyToCols(transformer, cols=selector)`` applies a
       transformer to selected columns
    3. **In DataOps:** ``skrub.X(df).skb.apply(transformer, cols=selector)``
    4. **Manual expansion:** ``selector.expand(df)`` gets column names for manual use

    **Combining Selectors**

    Selectors can be combined with operators to create complex selection rules:

    - ``s.numeric() | s.boolean()`` - numeric OR boolean columns
    - ``s.all() - s.glob('*_id')`` - all columns except ID-like ones
    - ``s.string() & ~s.cardinality_below(10)`` - high-cardinality string columns

    **Why Delayed Selection Matters**

    Selectors enable pipelines that work with different datasets:

    ```python
    # Without selectors: hardcoded column names break with different data
    transformer = ApplyToCols(StandardScaler(), cols=['height', 'width'])

    # With selectors: rules adapt to any dataset
    transformer = ApplyToCols(StandardScaler(), cols=s.numeric())
    ```

    See Also
    --------
    skrub.selectors : Module containing all available selector functions

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

    Use a selector to get matching column names:

    >>> s.numeric().expand(df)
    ['height_mm', 'width_mm', 'ID']

    Combine selectors for complex rules:

    >>> (s.numeric() & ~s.glob('*_ID')).expand(df)
    ['height_mm', 'width_mm']

    Use in data transformations:

    >>> from skrub import ApplyToCols
    >>> from sklearn.preprocessing import StandardScaler
    >>> ApplyToCols(StandardScaler(), cols=s.numeric()).fit_transform(df)
       height_mm  width_mm  ID  kind
    0       -1.0      -1.0   4    A4
    1        1.0       1.0   3    A3
    """

    def _matches(self, col):
        """Check if a column should be selected.

        This internal method is called by :meth:`expand` for each column in the
        dataframe. Subclasses must override this method to define their selection
        criteria.

        Parameters
        ----------
        col : column object
            A column from the dataframe (pandas Series, polars Series, etc.).

        Returns
        -------
        bool
            ``True`` if the column should be selected, ``False`` otherwise.

        Notes
        -----
        This is an internal method intended for subclass implementation.
        Users should use the public API functions like :func:`skrub.selectors.numeric`
        or :func:`skrub.selectors.glob` instead.
        """
        raise NotImplementedError()

    def expand(self, df):
        """Get the list of column names that the selector would select.

        This method evaluates the selector's matching criteria against each column
        in the dataframe and returns the names of columns that match.

        Use this method for exploratory work or when you need just the column names.
        For pipelines and transformers, pass the selector directly to
        :class:`~skrub.ApplyToCols`, :class:`~skrub.SelectCols`, etc.

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

        Use with dataframe selection:

        >>> df[some_selector.expand(df)]
           kind  ID
        0   A5   5
        1   A4   4

        Notes
        -----
        Internally, this method calls ``_matches(col)`` for each column. The selector
        is evaluated independently for each column, allowing for parallelization
        opportunities in the future.

        """
        matching_col_names = []
        for col_name in sbd.column_names(df):
            col = sbd.col(df, col_name)
            if self._matches(col):
                matching_col_names.append(col_name)
        return matching_col_names

    def expand_index(self, df):
        """Get the indices of columns that the selector would select.

        This method evaluates the selector against each column and returns the
        positional indices (0, 1, 2, ...) of matching columns instead of their names.

        Use this when you need column positions (e.g., for numpy-based operations
        or when column order matters more than names).

        Parameters
        ----------
        df : dataframe
            A pandas or polars dataframe to evaluate the selector against.

        Returns
        -------
        list of int
            The indices (0-based positions) of columns from ``df`` that the
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

        Use with column access by position:

        >>> cols = df.columns[[2, 3]]
        >>> df[cols]
           kind  ID
        0   A5   5
        1   A4   4

        Notes
        -----
        If ``cols`` is the list of column names in ``df``, then selecting by index
        and by name gives equivalent results:

        ``df[cols[i] for i in sel.expand_index(df)]`` == ``df[sel.expand(df)]``

        This method uses the same matching logic as :meth:`expand`; only the return
        format differs (indices vs. names).

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

    This is the most flexible selector, allowing arbitrary column selection logic
    based on column data (values, dtype, shape, etc.). The predicate receives the
    actual column object (pandas/polars Series) for inspection.

    When to use
    -----------
    - When built-in selectors don't match your selection criterion
    - To select columns based on column content properties (e.g., variance, range)
    - To select columns by custom statistics or computed properties
    - To combine multiple criteria in a single predicate
    - See ``filter_names`` for selections based only on column names

    Parameters
    ----------
    predicate : callable
        A function that takes a column and optional extra arguments, returning
        ``True`` to select the column or ``False`` to exclude it.
        Signature: ``predicate(col, *args, **kwargs) -> bool``
    *args : tuple
        Extra positional arguments passed to the predicate. Using explicit
        arguments (instead of closures) helps with pickling the selector.
    **kwargs : dict
        Extra keyword arguments passed to the predicate. Using explicit
        arguments (instead of closures) helps with pickling the selector.

    Returns
    -------
    Selector
        A ``Filter`` selector that matches columns where ``predicate`` returns ``True``.

    See Also
    --------
    filter_names : Select columns based only on their name (not content)
    Selector.expand : Evaluate a selector to get matching column names

    Notes
    -----
    **Predicate signature**: For a column ``col``, the predicate is called as::

        predicate(col, *args, **kwargs)

    The column is kept if the result is ``True``.

    **Pickling**: To pickle the selector, use importable functions as predicates
    rather than lambdas or closures. Pass parameters via ``*args`` or ``**kwargs``
    instead of capturing them in a closure::

        # Picklable (importable function + explicit args)
        s.filter(str.startswith, 'prefix')

        # Picklable alternative (with functools.partial)
        import functools
        s.filter(functools.partial(str.startswith, 'prefix'))

        # NOT picklable (closure)
        prefix = 'test'
        s.filter(lambda col: str(col.name).startswith(prefix))

    **Performance**: The predicate is called once per column, so operations on
    entire columns (not individual values) are efficient.

    Operator combinations
    ~~~~~~~~~~~~~~~~~~~~~
    - ``s.filter(lambda col: col.std() > 1) & s.numeric()`` → Numeric cols with high
      variance
    - ``s.filter(lambda col: len(col) > 100) | s.string()`` → Wide columns or strings
    - ``~s.filter(lambda col: col.isna().all())`` → All non-empty columns
    - ``s.filter(lambda col: col.dtype in [int, float]) - 'ID'`` → Numeric but not ID

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

    Select columns with high variance:

    >>> s.select(df, s.filter(lambda col: col.std() > 50))
       height_mm  width_mm
    0      297.0     210.0
    1      420.0     297.0

    Combine with type selectors:

    >>> def has_high_range(col):
    ...     return col.max() - col.min() > 100

    >>> s.select(df, s.numeric() & s.filter(has_high_range))
       height_mm  width_mm
    0      297.0     210.0
    1      420.0     297.0
    """
    return Filter(predicate, args=args, kwargs=kwargs)


class NameFilter(Filter):
    def _matches(self, col):
        return self.predicate(sbd.name(col), *self.args, **self.kwargs)

    @staticmethod
    def _default_name():
        return "filter_names"


def filter_names(predicate, *args, **kwargs):
    r"""Select columns based only on their name (not content or type).

    This is a specialized version of ``filter`` that passes only the column name
    (a string) to the predicate, rather than the column itself. Use this when
    your selection logic depends only on naming patterns, not data values.

    When to use
    -----------
    - Select columns matching regex patterns (use ``glob()`` or ``regex()``
      for simple patterns)
    - Check column names for prefixes, suffixes, or substrings
    - Filter by naming conventions (e.g., 'internal_*', 'tmp_*')
    - Custom naming-based logic that doesn't involve column data

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
        A ``NameFilter`` selector that matches columns where ``predicate(name)``
        returns ``True``.

    See Also
    --------
    filter : Select columns based on their content or properties
    glob : Select columns by wildcard pattern matching
    regex : Select columns by regular expression pattern matching
    Selector.expand : Evaluate a selector to get matching column names

    Notes
    -----
    **Key difference from ``filter``**: ``filter_names`` passes the column NAME
    (a string), while ``filter`` passes the column OBJECT (Series/DataFrame)::

        # filter receives the Series/array
        s.filter(lambda col: col.max() > 100)  # col is the Series

        # filter_names receives the string name
        s.filter_names(lambda name: name.startswith('x'))  # name is a string

    **Built-in alternatives**: For common patterns, prefer simpler selectors:

    - ``s.glob('*_mm')`` instead of ``s.filter_names(lambda n: n.endswith('_mm'))``
    - ``s.regex(r'col_\d+')`` instead of ``s.filter_names(lambda n: re.match(...))``

    **Pickling**: To pickle the selector, use importable functions as predicates
    rather than lambdas or closures. Pass parameters via ``*args`` or ``**kwargs``::

        # Picklable (importable function + explicit args)
        s.filter_names(str.endswith, '_mm')

        # Picklable alternative (with functools.partial)
        import functools
        s.filter_names(functools.partial(str.endswith, '_mm'))

        # NOT picklable (closure)
        suffix = '_mm'
        s.filter_names(lambda name: name.endswith(suffix))

    **Performance**: Very fast since only column names are inspected, no column
    data is loaded.

    Operator combinations
    ~~~~~~~~~~~~~~~~~~~~~
    - ``s.filter_names(str.startswith, 'in_') & s.numeric()`` → Internal numeric cols
    - ``s.filter_names(str.isupper) | s.glob('*_ID')`` → All caps names or *_ID cols
    - ``s.all() - s.filter_names(str.startswith, '_')`` → All except private cols
    - ``s.filter_names(lambda n: len(n) < 10)`` → Columns with short names

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

    Drop internal/private columns (starting with underscore):

    >>> s.drop(df, s.filter_names(str.startswith, '_'))
       height_mm  width_mm kind  ID
    0      297.0     210.0   A4   4
    1      420.0     297.0   A3   3
    """
    return NameFilter(predicate, args=args, kwargs=kwargs)
