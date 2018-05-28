=================================================
dirty_cat: machine learning on dirty categories
=================================================

.. toctree::
   :maxdepth: 2

.. currentmodule:: dirty_cat

`dirty_cat` is a small Python module to perform machine-learning on
non-curated categories. In particular, it provides encoders that are
robust to morphological variants, such as typos, in the category strings.

The :class:`SimilarityEncoder` can be used as a drop-in replacement for
the `scikit-learn <https://scikit-learn.org>`_ class
:class:`sklearn.preprocessing.OneHotEncoder`.

API documentation
=================


Base classes
------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SimilarityEncoder
   TargetEncoder

Datasets for examples
-----------------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.fetching.fetch_employee_salaries


.. include:: auto_examples/index.rst
    :start-line: 2

