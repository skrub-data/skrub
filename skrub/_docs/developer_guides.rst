New Developer Guides
====================

Welcome and thank you for your contribution to skrub
development!

Within the skrub library, we use a few internal modules
that make the skrub codebase easier to read and more robust.
Because they are specific to skrub and can be a little
opaque at first, we've included these guides to help new
developers get started using them.

Below, we provide introductions to the two main developer
modules, ``sbd`` and ``df_module``.

Using ``sbd`` in production code
---------------------------------

The first module, ``sbd``, provides backend-agnostic
dataframe functions for use in skrub's user-facing code
(instead of writing separate code for pandas and polars).

.. toctree::

    development/dataframe_api

Using ``df_module`` in tests
-----------------------------

The second module, ``df_module``, is mostly used when
writing test functions (a critical component of software
development!).

.. toctree::

    development/testing

Choosing between ``sbd`` and ``df_module``
-------------------------------------------

A comparison of the two modules and when
they are most used is also provided.

.. toctree::

    development/api_vs_testing
