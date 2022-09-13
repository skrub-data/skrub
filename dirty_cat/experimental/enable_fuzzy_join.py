"""Enables fuzzy_join function to be loaded.
# Inspired from https://github.com/scikit-learn/scikit-learn/blob/0.23.X/sklearn/experimental/enable_hist_gradient_boosting.py
The API and results of these function might change without any deprecation
cycle.
Importing this file dynamically sets the
:func:`dirty_cat.fuzzy_join` as attribute of the
dirty_cat module::
    >>> # explicitly require this experimental feature
    >>> from dirty_cat.experimental import enable_fuzzy_join  # noqa
    >>> # now you can import normally from dirty_cat
    >>> from dirty_cat import fuzzy_join
The ``# noqa`` comment comment can be removed: it just tells linters like
flake8 to ignore the import, which appears as unused.
"""

from dirty_cat._fuzzy_join import fuzzy_join

import dirty_cat

# use settattr to avoid mypy errors when monkeypatching
setattr(dirty_cat, "fuzzy_join",
        fuzzy_join)

dirty_cat.__all__ += ["fuzzy_join"]
