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

For a detailed description of the problem of encoding dirty categorical data,
see the article `Similarity encoding for learning with dirty categorical
variables <https://hal.inria.fr/hal-01806175>` [1]_.

.. [1] Patricio Cerda, Gaël Varoquaux, Balázs Kégl. Similarity encoding for learning with dirty categorical variables. 2018. Accepted for publication at: Machine Learning journal, Springer.

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

   datasets.fetch_midwest_survey
   datasets.fetch_employee_salaries


.. include:: auto_examples/index.rst
    :start-line: 2

