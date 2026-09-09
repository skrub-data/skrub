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
   :depth: 2


Why a parametrised fixture
---------------------------

skrub supports three distinct configurations in practice:

* **pandas with NumPy dtypes** (``pandas-numpy-dtypes``): the classical pandas
  dtypes backed by NumPy arrays — e.g. ``np.float64``, ``np.int64``.  Integer
  columns that contain ``None`` will be cast to ``float64`` because NumPy
  integers cannot represent missing values.

* **pandas with nullable extension dtypes** (``pandas-nullable-dtypes``):
  pandas' own nullable types — e.g. ``pd.Float64Dtype()``, ``pd.Int64Dtype()``.
  These represent missing values without promoting integer columns to float and
  behave somewhat differently from NumPy-backed dtypes.

* **polars** (``polars``): polars DataFrames and Series, which have their own
  type system, naming conventions, and API.

A test that only runs under one configuration may pass while silently failing
under the other two.  The ``df_module`` fixture ensures that a single test
function covers all three automatically.

How pytest sees it: a test that requests ``df_module`` is collected once and
run three times, once per parameter, producing independent pass/fail results.
If polars is not installed, the polars parameter is absent and the test runs
twice.


Anatomy of ``df_module``
-------------------------

``df_module`` is defined in ``skrub/conftest.py`` and returns a
:class:`types.SimpleNamespace` with a consistent set of attributes.  The
attributes are designed to normalise the differences between libraries so test
bodies need no ``if pandas / if polars`` branches (with a few exceptions).

The fixture signature:

.. code-block:: python

    @pytest.fixture(params=["pandas-numpy-dtypes", "pandas-nullable-dtypes", "polars"])
    def df_module(request):
        return _DATAFRAME_MODULES_INFO[request.param]

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

    +----------+------------------+-------------------+-------------+
    | Key      | numpy-dtypes     | nullable-dtypes   | polars      |
    +==========+==================+===================+=============+
    | float32  | ``np.float32``   | ``Float32Dtype``  | ``pl.Float32`` |
    +----------+------------------+-------------------+-------------+
    | float64  | ``np.float64``   | ``Float64Dtype``  | ``pl.Float64`` |
    +----------+------------------+-------------------+-------------+
    | int32    | ``np.int32``     | ``Int32Dtype``    | ``pl.Int32`` |
    +----------+------------------+-------------------+-------------+
    | int64    | ``np.int64``     | ``Int64Dtype``    | ``pl.Int64`` |
    +----------+------------------+-------------------+-------------+
    | category | ``CategoricalDtype`` | ``CategoricalDtype`` | ``pl.Categorical`` |
    +----------+------------------+-------------------+-------------+


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

Using the ``dtypes`` dict
~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose you are testing a function that should return a float32 column:

.. code-block:: python

    def test_returns_float32(df_module):
        col = df_module.make_column("x", [1, 2, 3])
        result = sbd.to_float32(col)
        assert sbd.dtype(result) == df_module.dtypes["float32"]


Defining example dataframes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Depending on the test, it is normally better to define dataframes that are tailored
for the desired test result. ``df_module.make_dataframe`` generates a dataframe for
each module starting from a python dictionary:

.. code-block:: python

    def test_column_names(df_module):
        df = df_module.make_dataframe({"a": [1, 2, 3], "b": [4, 5, 6]})
        names = sbd.column_names(df)
        assert "a" in names
        assert "b" in names

If multiple types are required for a given test, then ``example_dataframe`` can
be used to avoid boilerplate:

.. code-block:: python

    def test_column_names(df_module):
        df = df_module.example_dataframe
        names = sbd.column_names(df)
        assert "float-col" in names
        assert "datetime-col" in names


Related fixtures
-----------------

Several narrower fixtures complement ``df_module``. These fixtures have more niche
applications and are less common through the codebase.

``pd_module``
~~~~~~~~~~~~~

Always the ``"pandas-numpy-dtypes"`` configuration.  Use when you need to test
pandas-specific behaviour that is not part of the cross-backend API, or when
the test only makes sense for pandas.

.. code-block:: python

    def test_pandas_index_is_reset(pd_module):
        df = pd_module.make_dataframe({"a": [1, 2, 3]})
        df.index = [10, 20, 30]
        result = sbd.reset_index(df)
        assert list(result.index) == [0, 1, 2]

``pl_module``
~~~~~~~~~~~~~

The polars configuration.  If polars is not installed, the test is
automatically skipped with ``pytest.skip``.  Use for polars-specific
behaviour.

.. code-block:: python

    def test_lazyframe_is_rejected(pl_module):
        lazy = pl_module.empty_lazyframe
        with pytest.raises(TypeError, match="LazyFrames are not yet supported"):
            sbd.shape(lazy)

``all_dataframe_modules``
~~~~~~~~~~~~~~~~~~~~~~~~~

Returns the full ``dict`` mapping configuration name to namespace.  Use when
you need to iterate over all configurations programmatically inside a single
test body rather than through pytest parametrisation.

``use_fit_transform``
~~~~~~~~~~~~~~~~~~~~~

A boolean fixture parametrised as ``[False, True]``.  Use it to run the same
test through both ``fit`` + ``transform`` and ``fit_transform`` without
duplicating the test body:

.. code-block:: python

    def test_encoder(df_module, use_fit_transform):
        enc = MyEncoder()
        if use_fit_transform:
            result = enc.fit_transform(df_module.example_dataframe)
        else:
            result = enc.fit(df_module.example_dataframe).transform(
                df_module.example_dataframe
            )
        ...


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
