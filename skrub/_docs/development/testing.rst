.. _dev_testing:

Testing with the ``df_module`` fixture
=======================================

skrub's test suite must verify that every dataframe-aware feature works
correctly for all supported backends and dtype configurations.  Writing the
same test multiple times would be tedious and error-prone, so the suite provides a
parametrised fixture, ``df_module``, that multiplies a single test across all
configurations automatically.

This guide explains how ``df_module`` works, what attributes it provides, and
how to write effective tests using it.

.. contents:: Contents
   :local:
   :depth: 1

Anatomy of ``df_module``
-------------------------

``df_module`` is defined in ``skrub/conftest.py`` and returns a
:class:`types.SimpleNamespace` with a consistent set of attributes.  The
attributes are designed to normalise the differences between libraries so test
bodies need no ``if pandas / if polars`` branches (with a few exceptions).

Attributes
~~~~~~~~~~

``name`` — ``str``
    The library name: ``"pandas"`` or ``"polars"``.  Useful when a test must
    assert on the name or skip/branch based on the library.

    .. code-block:: python

        assert sbd.dataframe_module_name(df) == df_module.name

``description`` — ``str``
    The full configuration key: ``"pandas-numpy-dtypes"``,
    ``"pandas-nullable-dtypes"``, or ``"polars"``.  Use this when you need to
    distinguish between the two pandas configurations.

``module`` — module object
    The backend module itself (``pandas`` or ``polars``).  Useful if you need
    to access constants or secondary helpers directly.

``DataFrame`` — class
    The DataFrame class for this configuration: ``pd.DataFrame`` or
    ``pl.DataFrame``.

``Column`` — class
    The column/series class: ``pd.Series`` or ``pl.Series``.

``make_dataframe(data: dict) → DataFrame``
    Build a DataFrame from a column-name → values dictionary.  Under
    ``pandas-nullable-dtypes`` it additionally calls ``.convert_dtypes()``, so
    the resulting dtypes are the nullable extension types.

    .. code-block:: python

        df = df_module.make_dataframe({"a": [1, 2, 3], "b": ["x", "y", "z"]})

``make_column(name: str, values: list) → Column``
    Build a single column.

    .. code-block:: python

        col = df_module.make_column("score", [1.0, 2.5, None])

``assert_frame_equal(left, right, **kwargs)``
    Assert that two DataFrames are equal, using the backend's own testing
    helper (``pandas.testing.assert_frame_equal`` or
    ``polars.testing.assert_frame_equal``).

``assert_column_equal(left, right, **kwargs)``
    Assert that two columns are equal.

``empty_dataframe`` — DataFrame
    A DataFrame with zero rows and zero columns.  Useful as a trivial input to
    check that functions handle empty frames gracefully.

``empty_column`` — Column
    A column of length zero.

``empty_lazyframe`` — polars LazyFrame
    A lazy DataFrame with zero rows and zero columns.  **Only present for the
    polars configuration**; accessing it on a pandas ``df_module`` will raise
    ``AttributeError``.

``example_dataframe`` — DataFrame
    A ready-made DataFrame containing one column of each common dtype: integer
    (with nulls), integer (without nulls), float, string, boolean (with nulls),
    boolean (without nulls), datetime, and date.  The exact values are defined
    by ``_example_data_dict`` in ``conftest.py``.  Use this when you want a
    realistic multi-type frame without constructing one manually.

``example_column`` — Column
    The ``"float-col"`` column from ``example_dataframe`` (floats with one
    ``None``).

``dtypes`` — ``dict``
    A mapping from dtype name (string) to the appropriate dtype value for this
    configuration.  The keys are ``"float32"``, ``"float64"``, ``"int32"``,
    ``"int64"``, and ``"category"``.


Writing a basic test
---------------------

Here is a minimal test of a hypothetical ``my_transform`` function:

.. code-block:: python

    import skrub._dataframe as sbd

    def test_my_transform(df_module):
        # Build backend-appropriate inputs
        col = df_module.make_column("x", [1.0, 2.0, None, 4.0])

        result = my_transform(col)

        expected = df_module.make_column("x", [1.0, 4.0, None, 16.0])
        df_module.assert_column_equal(result, expected)

A few rules of thumb:

* Build inputs with ``df_module.make_dataframe`` / ``df_module.make_column``
  so that dtypes are correct for the current configuration.
* Assert with ``df_module.assert_frame_equal`` / ``df_module.assert_column_equal``
  rather than hand-rolling equality checks.  These helpers understand
  backend-specific equality semantics (e.g. null handling).
* When you need to check a dtype, use ``df_module.dtypes["float64"]`` rather
  than hard-coding ``np.float64``; the correct value depends on the
  configuration.


Polars-specific considerations
-------------------------------

LazyFrames
~~~~~~~~~~

The ``df_module`` fixture provides an ``empty_lazyframe`` attribute only for
the polars configuration. skrub functions expect an *eager* DataFrame;
passing a LazyFrame raises a ``TypeError`` with a message telling the caller
to call ``.collect()``.  Test this behaviour explicitly if your function could
receive a LazyFrame.

The ``skip_polars_installed_without_pyarrow`` mark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For some operations (some date/time conversions, functions that involve the
computation of column associations) it is not possible to rely exclusively on polars,
because some features are not available; in such cases, the dataframe is silently
converted to pandas and then back to polars when possible. This conversion requires
the pyarrow package, which is an optional dependency.
A pytest mark is available to skip tests for these operations when polars is
installed but pyarrow is not:

.. code-block:: python

    from skrub.conftest import skip_polars_installed_without_pyarrow

    @skip_polars_installed_without_pyarrow
    def test_datetime_conversion(df_module):
        ...

Apply this mark to tests that call polars functionality backed by pyarrow.

A specific CI environment (``ci-py314-polars-without-pyarrow``) is used to test
this situation.

Where tests live
-----------------

Tests for transformers and their functions live in their respective test file:
the code of the ``DatetimeEncoder`` is in ``skrub/_datetime_encoder.py``, while
its tests are in ``skrub/tests/test_datetime_encoder.py``.

Each submodule contains both the code and its tests. For example, the code for the
dataframe API is in ``skrub/_dataframe``, while the relative tests are in
``skrub/_dataframe/tests``. All tests can request ``df_module``: the fixture is
visible to the entire ``skrub/`` test tree because it is defined in
``skrub/conftest.py``.
