.. _dev_api_vs_testing:

Comparing the dispatch API and ``df_module``
============================================================

skrub has two distinct mechanisms for handling multiple dataframe backends:
the dispatched *dataframe API* (``skrub/_dataframe``, ``skrub/_dispatch.py``) and
the *``df_module`` fixture* (``skrub/conftest.py``). They look superficially
similar — both abstract over pandas and polars — but they exist at different
levels and serve different roles.

+---------------------------+--------------------------------------+------------------------------------+
|                           | Dispatch API                         | ``df_module`` fixture              |
+===========================+======================================+====================================+
| **Where**                 | ``skrub/_dataframe/_common.py``,     | ``skrub/conftest.py``              |
+---------------------------+--------------------------------------+------------------------------------+
| **When it runs**          | Production — at import time and      | Test time — under pytest only      |
|                           | when skrub functions are called      |                                    |
+---------------------------+--------------------------------------+------------------------------------+
| **What it abstracts**     | *How* to perform an operation        | *How* to construct inputs and      |
|                           | (fill nulls, get shape, cast, ...)   | assert outputs in a test           |
+---------------------------+--------------------------------------+------------------------------------+


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
