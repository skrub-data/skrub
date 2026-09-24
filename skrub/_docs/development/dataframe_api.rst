.. _dev_dataframe_api:

The dispatch-based dataframe API
=================================

skrub targets both pandas and polars as first-class backends.  Rather than
scattering ``if pandas ... else polars ...`` branches throughout the codebase, all
dataframe and column operations are funneled through a thin dispatch layer that
selects the right implementation at call time.  This guide explains how that
layer works and how to extend it.

.. contents:: Contents
   :local:
   :depth: 1

How dispatching works
---------------------

The mechanism lives in ``skrub/_dispatch.py`` and is built on top of the
standard library's :func:`functools.singledispatch`.

``functools.singledispatch`` selects an implementation based on the *type* of
the first argument, dealing with the fact that polars is not a required dependency
internally.

The ``@dispatch`` decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Implementing library-specific code with ``specialize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Using the API
-------------

Import convention
~~~~~~~~~~~~~~~~~

Throughout skrub, the module is imported under the alias ``sbd`` (or
occasionally ``ns`` in older code and docstrings):

.. code-block:: python

    import skrub._dataframe as sbd

This is a private module; it is not part of the public skrub API.

All public functions are re-exported from ``skrub/_dataframe/__init__.py``
via ``from ._common import *``.  They are grouped conceptually in
``_common.__all__``.

Example usage:

.. code-block:: python

    import skrub._dataframe as sbd

    # Works for a pandas Series or a polars Series
    col = ...
    if sbd.has_nulls(col):
        col = sbd.fill_nulls(col, 0)

    # Works for a pandas DataFrame or a polars DataFrame
    df = ...
    n_rows, n_cols = sbd.shape(df)
    names = sbd.column_names(df)


Adding a function to ``_common.py``
------------------------------------

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


Defining dispatched functions outside ``_common.py``
------------------------------------------------------

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
