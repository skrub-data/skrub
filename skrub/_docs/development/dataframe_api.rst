.. _dev_dataframe_api:

The dispatch-based dataframe API
=================================

skrub targets both pandas and polars as first-class backends.  Rather than
scattering ``if pandas … else polars …`` branches throughout the codebase, all
dataframe and column operations are funneled through a thin dispatch layer that
selects the right implementation at call time.  This guide explains how that
layer works and how to extend it.

.. contents:: Contents
   :local:
   :depth: 2


Motivation
----------

The naive approach to multi-backend support is to branch on the library name
wherever an operation is performed:

.. code-block:: python

    # don't do this
    if isinstance(col, pd.Series):
        result = col.fillna(0)
    else:
        result = col.fill_null(0)

This pattern is fragile: the same check must be repeated everywhere, adding a
third backend means touching every branch, and tests must cover every path
explicitly.  The dispatch layer inverts the design: each operation is defined
once as a *generic function*, and the concrete implementation is registered
separately per library.  Call sites stay library-agnostic.

.. code-block:: python

    import skrub._dataframe as sbd

    result = sbd.fill_nulls(col, 0)   # works for pandas Series or polars Series


How dispatching works
---------------------

The mechanism lives in ``skrub/_dispatch.py`` and is built on top of the
standard library's :func:`functools.singledispatch`.

``functools.singledispatch`` selects an implementation based on the *type* of
the first argument.  The wrinkle is that some backends (currently polars) are
optional dependencies: you cannot import ``polars.DataFrame`` to register a
specialisation if polars is not installed.  The ``dispatch`` decorator works
around this by accepting the library name as a string and resolving types only
when the library is actually importable.

The type registry
~~~~~~~~~~~~~~~~~

Internally, ``_dispatch.py`` maintains a registry mapping library names to
their concrete types:

.. code-block:: text

    "pandas" → {
        "DataFrame": (pandas.DataFrame,),
        "Column":    (pandas.Series,),
    }

    "polars" → {
        "DataFrame": (polars.DataFrame,),
        "LazyFrame": (polars.LazyFrame,),
        "EagerFrame": (polars.DataFrame,),
        "Column":    (polars.Series,),
    }

These string names (``"DataFrame"``, ``"Column"``, ``"LazyFrame"``) are what
you pass to ``specialize``'s ``argument_type`` keyword.

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

Registering specialisations with ``specialize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @fill_nulls.specialize("pandas", argument_type="Column")
    def _fill_nulls_pandas(col, value):
        return col.fillna(value)

    @fill_nulls.specialize("polars", argument_type="Column")
    def _fill_nulls_polars(col, value):
        return col.fill_null(value)

``specialize`` takes two arguments:

* **Library name** (the strings ``"pandas"`` or ``"polars"``): the concrete
  implementations are looked up only if that library is importable; otherwise
  the decorator is a no-op and the function is never registered.
* **``argument_type``** (optional): one of the string keys in the type registry,
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
priority ordering based on specificity.  This means that if you register a
specialisation for all pandas types and then later register one for
``"Column"`` only, the second one will override the first for Series objects.

Naming convention
~~~~~~~~~~~~~~~~~

Specialised implementations follow the pattern
``_<generic_name>_<library>`` for library-wide specialisations, or
``_<generic_name>_<library>_<type>`` when the ``argument_type`` is scoped:

.. code-block:: python

    _fill_nulls_pandas          # pandas-wide
    _fill_nulls_polars          # polars-wide
    _to_numpy_pandas_column     # pandas, Column only
    _to_numpy_pandas_table      # pandas, DataFrame only

The names are arbitrary and have no effect on dispatch; they are a
documentation and searchability convention.

Error messages
~~~~~~~~~~~~~~

``raise_dispatch_unregistered_type`` produces three distinct error messages:

* **Unknown type**: "Expecting a Pandas or Polars <kind>, but got …"
* **DataOp**: tells the caller to use ``.skb.eval()`` or ``.skb.apply_func()``
  instead.
* **LazyFrame**: tells the caller to call ``.collect()`` first.


Using the API
-------------

Import convention
~~~~~~~~~~~~~~~~~

Throughout skrub, the module is imported under the alias ``sbd`` (or
occasionally ``ns`` in older code and docstrings):

.. code-block:: python

    import skrub._dataframe as sbd

This is a private module; it is not part of the public skrub API.

Available functions
~~~~~~~~~~~~~~~~~~~

All public functions are re-exported from ``skrub/_dataframe/__init__.py``
via ``from ._common import *``.  They are grouped conceptually in
``_common.__all__``:

**Type inspection**
    ``dataframe_module_name``, ``is_pandas``, ``is_polars``,
    ``is_dataframe``, ``is_lazyframe``, ``is_column``

**Conversions**
    ``to_list``, ``to_numpy``, ``to_pandas``,
    ``make_dataframe_like``, ``make_column_like``, ``null_value_for``,
    ``all_null_like``, ``concat``, ``is_column_list``, ``to_column_list``,
    ``col``, ``col_by_idx``, ``collect``

**Shape and metadata**
    ``shape``, ``to_frame``, ``name``, ``column_names``, ``rename``,
    ``set_column_names``, ``reset_index``, ``copy_index``, ``index``, ``drop``

**Dtype inspection and casting**
    ``dtype``, ``dtypes``, ``cast``, ``is_bool``, ``is_numeric``,
    ``is_integer``, ``is_float``, ``to_float32``, ``is_string``, ``to_string``,
    ``is_object``, ``is_any_date``, ``to_datetime``, ``is_duration``,
    ``is_categorical``, ``to_categorical``, ``is_all_null``, ``is_empty_frame``,
    ``is_pandas_extension_dtype``, ``pandas_convert_dtypes``, ``is_pandas_object``

**Values**
    ``all``, ``any``, ``sum``, ``min``, ``max``, ``std``, ``mean``,
    ``pearson_corr``, ``sort``, ``value_counts``, ``quantile``,
    ``is_null``, ``has_nulls``, ``drop_nulls``, ``fill_nulls``,
    ``n_unique``, ``unique``, ``filter``, ``where``, ``where_row``,
    ``sample``, ``head``, ``slice``, ``select_rows``, ``replace``,
    ``with_columns``, ``abs``, ``total_seconds``, ``is_sorted``

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

Step 1 — write the generic function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the function near related ones in ``skrub/_dataframe/_common.py``.
The first argument must be the dataframe or column that will drive dispatch.

.. code-block:: python

    @dispatch
    def clip(col, lower, upper):
        """Clip values in a column to [lower, upper]."""
        raise_dispatch_unregistered_type(col, kind="Series")

If a sensible no-op default exists (e.g. the operation is pandas-specific),
you can return ``obj`` or another safe value instead of raising.

Step 2 — add specialisations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @clip.specialize("pandas", argument_type="Column")
    def _clip_pandas(col, lower, upper):
        return col.clip(lower=lower, upper=upper)

    @clip.specialize("polars", argument_type="Column")
    def _clip_polars(col, lower, upper):
        return col.clip(lower_bound=lower, upper_bound=upper)

Step 3 — add to ``__all__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the function name to the ``__all__`` list at the top of ``_common.py``,
in the appropriate section.

Step 4 — write tests
~~~~~~~~~~~~~~~~~~~~~

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

The pattern is identical to ``_common.py``, using the same ``dispatch`` and
``raise_dispatch_unregistered_type`` imported from ``skrub._dispatch``:

.. code-block:: python

    # skrub/_my_transformer.py

    from ._dispatch import dispatch, raise_dispatch_unregistered_type
    import skrub._dataframe as sbd

    @dispatch
    def _extract_something(col):
        raise_dispatch_unregistered_type(col, kind="Series")

    @_extract_something.specialize("pandas", argument_type="Column")
    def _extract_something_pandas(col):
        # pandas-specific implementation
        return col.str.extract(r"(\d+)")

    @_extract_something.specialize("polars", argument_type="Column")
    def _extract_something_polars(col):
        # polars-specific implementation
        return col.str.extract(r"(\d+)", group_index=0)

A real example from ``skrub/_datetime_encoder.py``:

.. code-block:: python

    @dispatch
    def _is_date(col):
        from ._dispatch import raise_dispatch_unregistered_type
        raise_dispatch_unregistered_type(col, kind="Series")

    @_is_date.specialize("pandas", argument_type="Column")
    def _is_date_pandas(col):
        col = sbd.drop_nulls(col)
        return (col.dt.normalize() == col).all()

    @_is_date.specialize("polars", argument_type="Column")
    def _is_date_polars(col):
        return (col.dt.date() == col).all()

Functions defined this way are **not** exported from ``skrub._dataframe``; they
are module-private helpers.  Only add a function to ``_common.py`` and its
``__all__`` when it is genuinely reusable across multiple parts of skrub.

Specialised dispatched functions may involve complex operations with multiple
steps depending on the situation (e.g., ``_session_encoder.py``): there is no need
to break the operations into smaller functions if this would result in chaining
multiple dispatched functions anyway.

Rules for production code
--------------------------

* **Always use** ``sbd.*`` for dataframe or column operations; never call
  ``df.method()`` directly outside a ``specialize`` block.
* Keep the **first argument** as the dispatching argument (the dataframe or
  column).  A function ``sample(n, df)`` would dispatch on ``n``, which is
  wrong; it must be ``sample(df, n)``.
* Inside a ``specialize`` block, it is safe to import the backend module and
  call any of its methods — you are guaranteed the first argument is that
  backend's type.
* Outside a ``specialize`` block, never import polars unconditionally; polars
  is an optional dependency.
