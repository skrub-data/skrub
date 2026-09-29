New Developer Guides
====================

Welcome and thank you for your contribution to skrub
development!

Within the skrub library, we have a few internal modules
that make the skrub codebase easier to read and more robust.
Because they are specific to skrub and can be a little
opaque at first, we've included these getting started
guides to introduce new developers to how we use skrub
internal modules to make skrub easier to use and maintain.

Below, we provide introductions to the two main developer
modules, ``sbd`` and ``df_module``. The first module,
``sbd``, provides backend-agnostic dataframe functions for
use in skrub's user-facing code (instead of writing separate
code for pandas and polars). The second module, ``df_module``, is mostly used when
writing test functions (a critical component of software
development!). A comparison of the two modules and when
they are most used is also provided.

.. toctree::

    development/dataframe_api
    development/testing
    development/api_vs_testing
