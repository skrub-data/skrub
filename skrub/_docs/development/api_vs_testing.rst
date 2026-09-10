.. _dev_api_vs_testing:

Two layers, one purpose: the dispatch API and ``df_module``
============================================================

skrub has two distinct mechanisms for handling multiple dataframe backends:
the dispatched *dataframe API* (``skrub/_dataframe``, ``skrub/_dispatch.py``) and
the *``df_module`` fixture* (``skrub/conftest.py``). They look superficially
similar — both abstract over pandas and polars — but they exist at different
levels and serve different roles.  This guide explains the design rationale,
where the boundary between them lies, and how to decide which to use.

.. contents:: Contents
   :local:
   :depth: 2


The two layers at a glance
---------------------------

+---------------------------+--------------------------------------+------------------------------------+
|                           | Dispatch API                         | ``df_module`` fixture              |
+===========================+======================================+====================================+
| **Where**                 | ``skrub/_dataframe/_common.py``,     | ``skrub/conftest.py``              |
|                           | ``skrub/_dispatch.py``               |                                    |
+---------------------------+--------------------------------------+------------------------------------+
| **When it runs**          | Production — at import time and      | Test time — under pytest only      |
|                           | when skrub functions are called      |                                    |
+---------------------------+--------------------------------------+------------------------------------+
| **What it abstracts**     | *How* to perform an operation        | *How* to construct inputs and      |
|                           | (fill nulls, get shape, cast, …)     | assert outputs in a test           |
+---------------------------+--------------------------------------+------------------------------------+
| **Mechanism**             | ``functools.singledispatch`` +       | pytest ``@fixture(params=…)``      |
|                           | ``specialize`` decorator             |                                    |
+---------------------------+--------------------------------------+------------------------------------+
| **Result of abstraction** | A single call site (``sbd.fill_nulls``) | A single test body runs 3 times |
|                           | works for any backend                | (one per configuration)            |
+---------------------------+--------------------------------------+------------------------------------+


The dataframe API: writing library-agnostic production code
-----------------------------------------------------------

The dataframe API solves a problem that arises *at runtime*: a transformer
receives a DataFrame or a Series whose backend is not known at the time the
code was written.  The ``@dispatch`` / ``specialize`` mechanism routes the
call to the correct implementation based on the actual type of the object.

Call sites in production code are completely library-agnostic:

.. code-block:: python

    import skrub._dataframe as sbd

    def _process(col):
        if sbd.has_nulls(col):
            col = sbd.fill_nulls(col, 0)
        return sbd.to_float32(col)

Neither ``pandas`` nor ``polars`` is imported here.  The correct
implementation — ``col.fillna(0)`` for pandas or ``col.fill_null(0)`` for
polars — is selected automatically.

The key properties of the dataframe API:

* It is **production code** that ships as part of skrub's package.
* It is imported and executed whenever a user calls a skrub estimator.
* It says nothing about *how to build* a DataFrame or *how to assert equality*;
  it only defines *operations* on existing objects.
* It must handle real user data — arbitrary DataFrames that arrive from outside
  skrub.


``df_module``: testing library-agnostic code
---------------------------------------------

``df_module`` solves a different problem: when a test needs to construct
inputs, call a function, and check the output, it must do so in a way that
works for pandas with numpy dtypes, pandas with nullable dtypes and polars (if
available).  ``df_module`` provides a uniform interface for these test-time concerns.

A test using ``df_module`` is collected once and run three times by pytest:

.. code-block:: python

    def test_fill_nulls(df_module):
        col = df_module.make_column("x", [1.0, None, 3.0])
        result = sbd.fill_nulls(col, 0.0)
        expected = df_module.make_column("x", [1.0, 0.0, 3.0])
        df_module.assert_column_equal(result, expected)

``df_module.make_column`` builds a ``pd.Series`` (numpy dtypes), a
``pd.Series`` (nullable dtypes), or a ``pl.Series`` depending on the
parameter.  ``df_module.assert_column_equal`` delegates to the right
``*.testing`` module.  No ``if`` branches, no duplication.

The key properties of ``df_module``:

* It is **test infrastructure** that lives in ``conftest.py`` and is never
  imported in production code.
* It is only active when pytest runs.
* It knows how to *construct* DataFrames and *assert equality*, not how to
  perform arbitrary operations on them.
* It works with controlled, synthetic data, not real user data.


Why not collapse them?
----------------------

The two layers could theoretically be merged — for example, ``df_module``
could use ``sbd.*`` to build its example objects.  There are deliberate
reasons not to do this.

**``df_module`` does not build on the dataframe API.**
    ``df_module`` is part of the test bootstrap.  If it relied on ``sbd.*``
    internally, a bug in the dispatch layer would corrupt the test inputs
    themselves, making it impossible to distinguish "the function under test is
    broken" from "the test fixture is broken".  Using the backends' own
    constructors (``pd.DataFrame.from_dict``, ``pl.from_dict``, …) keeps the
    fixture independent.

**The dataframe API does not know about test concerns.**
    The dispatch API is a production abstraction over dataframe *operations*.
    Concepts like "an example DataFrame with four rows and mixed dtypes" or
    "assert that two Series are equal up to dtype" are test concerns, not
    operation concerns.  Mixing them would blur the boundary between production
    and test code.

**Three configurations, not two libraries.**
    The dataframe API dispatches on ``pandas.DataFrame`` vs ``polars.DataFrame``
    — two cases.  The test suite has three configurations: pandas-numpy-dtypes,
    pandas-nullable-dtypes, and polars.  The extra pandas configuration catches
    dtype-specific bugs that would be invisible if tests only ran against one
    pandas variant.  The dataframe layer has no notion of "which pandas",
    because from a runtime perspective both are ``pd.DataFrame``; only the test
    layer needs to distinguish them.


Decision guide
--------------

Use this table to decide where new code or infrastructure belongs.

+-----------------------------------------------+-----------------------------------+
| Situation                                     | What to do                        |
+===============================================+===================================+
| I need to perform a dataframe operation in    | Use ``sbd.*``.  If the function   |
| a transformer or utility function.            | does not exist yet, add it to     |
|                                               | ``_common.py`` (see               |
|                                               | :ref:`dev_dataframe_api`).        |
+-----------------------------------------------+-----------------------------------+
| I need to perform an operation that is        | Define a local ``@dispatch``      |
| specific to one module (e.g. a helper in      | function in that module.  Do not  |
| ``_datetime_encoder.py``) and has no reuse    | add it to ``_common.py``.         |
| outside it.                                   |                                   |
+-----------------------------------------------+-----------------------------------+
| I need to write a test for code that touches  | Use ``df_module``.                |
| a DataFrame or column.                        |                                   |
+-----------------------------------------------+-----------------------------------+
| I need to construct a backend-appropriate     | Use ``df_module.make_dataframe``  |
| DataFrame in a test.                          | or ``df_module.make_column``.     |
+-----------------------------------------------+-----------------------------------+
| I need to assert equality in a test.          | Use                               |
|                                               | ``df_module.assert_frame_equal``  |
|                                               | or                                |
|                                               | ``df_module.assert_column_equal``.|
+-----------------------------------------------+-----------------------------------+
| I am unsure whether a new operation belongs   | If other transformers would       |
| in ``_common.py`` or should be local.         | benefit from it: ``_common.py``.  |
|                                               | If it is specific to one class:   |
|                                               | local ``@dispatch``.              |
+-----------------------------------------------+-----------------------------------+

A concrete heuristic: if you are writing code that will run when a user calls
``TableVectorizer().fit_transform(df)``, use the dataframe API.  If you are
writing code that only runs under ``pytest``, use ``df_module``.


Parallel structure
------------------

Despite their different purposes, the two layers do mirror each other in one
respect: both have a "all backends" path and a "specific backend" escape hatch.

In the **dataframe API**:

* ``sbd.fill_nulls(col, 0)`` — generic, works for any backend.
* ``@fill_nulls.specialize("pandas", argument_type="Column")`` — pandas-specific
  implementation, invoked automatically.

In the **test fixture**:

* ``df_module`` — generic, runs for every configuration.
* ``pd_module`` / ``pl_module`` — fixed to one backend, used when the test
  itself is backend-specific.

The design principle is the same in both cases: write the general case once
and isolate backend differences to dedicated, clearly labeled places.
