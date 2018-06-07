=================================================
dirty_cat: machine learning on dirty categories
=================================================

.. toctree::
   :maxdepth: 2

.. currentmodule:: dirty_cat

`dirty_cat` is a small Python module to perform machine-learning on
non-curated categories. In particular, it provides **encoders that are
robust to morphological variants**, such as typos, in the category strings.

The :class:`SimilarityEncoder` can be used as a drop-in replacement for
the `scikit-learn <https://scikit-learn.org>`_ class
:class:`sklearn.preprocessing.OneHotEncoder`.

For a detailed description of the problem of encoding dirty categorical data,
see `Similarity encoding for learning with dirty categorical variables
<https://hal.inria.fr/hal-01806175>`_ [1]_.

:Installing: `$ pip install --user dirty_cat`

______

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

   datasets.fetch_employee_salaries

.. include:: auto_examples/index.rst
    :start-line: 2

About
=========

dirty_cat is for now a repository for developing ideas with high-quality
implementations, a form of a research project: there is still
little known about the problems of dirty categories. We hope that
tradeoffs will emerge in the long run, and that these tradeoffs will
enable us to do better software. We really need people giving feedback on
success and failures with the different techniques and pointing us to
open datasets on which we can do more empirical work. We also welcome
contributions in the scope of dirty categories.


.. seealso::

   Many classic categorical encoding schemes are available here:
   http://contrib.scikit-learn.org/categorical-encoding/

.. [1] Patricio Cerda, Gaël Varoquaux, Balázs Kégl. Similarity encoding for learning with dirty categorical variables. 2018. Accepted for publication at: Machine Learning journal, Springer.

