.. _dev_dataframe_api:

The dispatch-based dataframe API
=================================

skrub can be used with both pandas and polars dataframes without
requiring users to change how they interact with the library.
That is to say that when you pass a dataframe to a skrub function,
it automatically detects which dataframe library (i.e. which backend) is
being used and provides equivalent behavior, even though pandas
and polars have different syntax (ie. their APIs). The user does not have
to think about which type of dataframe is being inputted, which
is all handled internally by the skrub library.

Now a simple way to implement this would be to have separate code
blocks in every skrub function for each condition
(e.g. ``if pandas ... else polars ...``) however this is quite
cumbersome, and is not super extensible (what if we wanted to add
support for another dataframe library?). Instead, all the library specific
behavior is encapsulated into a dispatch layer that selects the right
implementation at call time. For most functions, this will be done
using the ``sbd`` module.

This guide explains how to use these generic (or backend-agnostic)
``sbd`` functions to write robust skrub functions that work with
both pandas and polars.

.. contents:: Contents
   :local:
   :depth: 1

Using the dispatch API as ``sbd``
-------------

Throughout skrub, we use a common dataframe module so that all functions are
written with one internal implementation and where any backend-specific
behavior is handled within the dispatch layer. The most common usage is
to import the module under the alias ``sbd`` (or occasionally ``ns`` in
older code and docstrings):

.. code-block:: python

    import skrub._dataframe as sbd

This is a private module; it is not part of the public skrub API.

All public functions are re-exported from ``skrub/_dataframe/__init__.py``
via ``from ._common import *``.  They are grouped conceptually in
``_common.__all__``.

Once imported, the ``sbd`` module can be used to perform dataframe operations
such as getting column names, or checking the type of dataframe.

For example, compare the two approaches (first handling both pandas and
polars cases for each method) versus using ``sbd``:

.. code-block:: python

    # DISCOURAGED: messy imports required in each file
    import pandas as pd

    try:
        import polars as pl
    except ImportError:
        pl = None

    # PREFERRED: imports handled in single module
    import skrub._dataframe as sbd

    # DISCOURAGED: separate code blocks for each backend
    col = ...
    if isinstance(col, pd.Series):
        if col.isna().any():
            col = col.fillna(0)
    elif pl is not None and isinstance(col, pl.Series):
        if col.null_count() > 0:
            col = col.fill_null(0)
    else:
        raise TypeError("Unsupported column type")

    # PREFERRED: Works for a pandas Series or a polars Series
    if sbd.has_nulls(col):
        col = sbd.fill_nulls(col, 0)

    # DISCOURAGED: more conditional checks for each backend
    df = ...
    if isinstance(df, pd.DataFrame):
        n_rows, n_cols = df.shape
        names = df.columns.tolist()
    elif pl is not None and isinstance(df, pl.DataFrame):
        n_rows, n_cols = df.shape
        names = df.columns
    else:
        raise TypeError("Unsupported dataframe type")

    # PREFERRED: Works for a pandas DataFrame or a polars DataFrame
    n_rows, n_cols = sbd.shape(df)
    names = sbd.column_names(df)


The nitty gritty of dispatching
--------------------------------

The mechanism lives in ``skrub/_dispatch.py`` and is built on top of the
standard library's :func:`functools.singledispatch`.

``functools.singledispatch`` selects an implementation based on the *type* of
the first argument, dealing with the fact that polars is not a required dependency
internally.

**The ``@dispatch`` decorator**

Applying ``@dispatch`` to a function converts it into a generic function and
adds a ``specialize`` attribute:

.. code-block:: python

    from skrub._dispatch import dispatch, raise_dispatch_unregistered_type

    @dispatch
    def fill_nulls(col, value):
        raise_dispatch_unregistered_type(col, kind="Series")

The default body is the fallback that runs when no specialisation has been
registered for the argument's type.  The idiomatic choice is to raise a
descriptive error with ``raise_dispatch_unregistered_type`` when an unsupported
type is passed as an argument: for example, in case a pure python list is used
instead of a Pandas series.
In some cases, functions use a safe no-op default, rather than raising an exception
(e.g. ``reset_index`` which is a pandas concept and simply returns ``obj`` unchanged
for everything else).

**Implementing library-specific code with ``specialize``**

.. code-block:: python

    @fill_nulls.specialize("pandas", argument_type="Column")
    def _fill_nulls_pandas(col, value):
        return col.fillna(value)

    @fill_nulls.specialize("polars", argument_type="Column")
    def _fill_nulls_polars(col, value):
        return col.fill_null(value)

``specialize`` takes two arguments:

* **Library name** (the strings ``"pandas"`` or ``"polars"``).
* ``argument_type`` (optional): one of the string keys in the type registry,
  or a tuple of them.  Omitting it registers the specialisation for *all* types
  in that library (DataFrame, Column, and LazyFrame for polars).

+----------------------------------+------------------------------------------------+
| ``argument_type``                | Registers for                                  |
+==================================+================================================+
| ``None`` (default)               | All types in the library                       |
+----------------------------------+------------------------------------------------+
| ``"DataFrame"``                  | DataFrame class only                           |
+----------------------------------+------------------------------------------------+
| ``"Column"``                     | Series class only                              |
+----------------------------------+------------------------------------------------+
| ``"LazyFrame"``                  | polars LazyFrame only                          |
+----------------------------------+------------------------------------------------+
| ``("DataFrame", "Column")``      | Both DataFrame and Series                      |
+----------------------------------+------------------------------------------------+

The **last** registered specialisation wins for a given type; there is no
priority ordering based on specificity.

So you really want to add a new ``sbd`` function
-------------------------------------------------
**Adding a function to ``_common.py``**

**Step 1 — write the generic function**

Add the function near related ones in ``skrub/_dataframe/_common.py``.
The first argument must be the dataframe or column that will drive dispatch.

.. code-block:: python

    @dispatch
    def clip(col, lower, upper):
        """Clip values in a column to [lower, upper]."""
        raise_dispatch_unregistered_type(col, kind="Series")

If a sensible no-op default exists (e.g. the operation is pandas-specific),
you can return ``obj`` or another safe value instead of raising.

**Step 2 — add specialisations**

.. code-block:: python

    @clip.specialize("pandas", argument_type="Column")
    def _clip_pandas(col, lower, upper):
        return col.clip(lower=lower, upper=upper)

    @clip.specialize("polars", argument_type="Column")
    def _clip_polars(col, lower, upper):
        return col.clip(lower_bound=lower, upper_bound=upper)

**Step 3 — add to** ``__all__``

Add the function name to the ``__all__`` list at the top of ``_common.py``,
in the appropriate section.

**Step 4 — write tests**

Add a test in ``skrub/_dataframe/tests/test_common.py`` using the
``df_module`` fixture (see :ref:`dev_testing`).

.. code-block:: python

    def test_clip(df_module):
        col = df_module.make_column("x", [1, 5, 10, -3])
        result = sbd.clip(col, lower=0, upper=7)
        expected = df_module.make_column("x", [1, 5, 7, 0])
        df_module.assert_column_equal(result, expected)


**Defining dispatched functions outside ``_common.py``**

Not all dispatched functions belong in ``_common.py``.  If the operation is
tightly coupled to a specific transformer or sub-module and has no use
elsewhere, define it locally in that module.  Examples include
``_is_date`` and ``_get_dt_feature`` in ``skrub/_datetime_encoder.py``, and
``_str_replace`` in ``skrub/_to_float.py``.

Functions defined this way are **not** exported from ``skrub._dataframe``; they
are module-private helpers.  Only add a function to ``_common.py`` and its
``__all__`` when it is genuinely reusable across multiple parts of skrub.

Specialised dispatched functions may involve complex operations with multiple
steps depending on the situation (e.g., ``_session_encoder.py``): there is no need
to break the operations into smaller functions if this would result in chaining
multiple dispatched functions anyway.
