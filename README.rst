dirty_cat
=========

.. image:: https://dirty-cat.github.io/stable/_static/dirty_cat.svg
   :align: center
   :alt: dirty_cat logo


|py_ver| |pypi_var| |pypi_dl| |codecov| |circleci| |black|

.. |py_ver| image:: https://img.shields.io/pypi/pyversions/dirty_cat
.. |pypi_var| image:: https://img.shields.io/pypi/v/dirty_cat?color=informational
.. |pypi_dl| image:: https://img.shields.io/pypi/dm/dirty_cat
.. |codecov| image:: https://img.shields.io/codecov/c/github/dirty-cat/dirty_cat/master
.. |circleci| image:: https://img.shields.io/circleci/build/github/dirty-cat/dirty_cat/master?label=CircleCI
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg

`dirty_cat <https://dirty-cat.github.io/>`_ is a Python library
that facilitates machine-learning on dirty categorical variables.

For a detailed description of the problem of encoding dirty categorical data, see
`Similarity encoding for learning with dirty categorical variables <https://hal.inria.fr/hal-01806175>`_ [1]_
and `Encoding high-cardinality string categorical variables <https://hal.inria.fr/hal-02171256v4>`_ [2]_.

If you like the package, please *spread the word*, and ⭐ the repository!

What can dirty_cat do?
----------------------

dirty_cat provides encoders for **morphological similarities** in the data,
which means that it assumes a correlation between the similarities
in the feature we want to encode, and similarities in the target.

We usually identify three common cases: **similarities, typos and variations**

`Example 1 <https://dirty-cat.github.io/stable/auto_examples/01_dirty_categories.html>`_ goes into detail on how to identify and deal with dirty data.
using dirty_cat

The encoders and tools provided by dirty_cat include:

- The `SimilarityEncoder <https://dirty-cat.github.io/stable/generated/dirty_cat.SimilarityEncoder.html>`_
- The `GapEncoder <https://dirty-cat.github.io/stable/generated/dirty_cat.GapEncoder.html>`_
- The `MinHashEncoder <https://dirty-cat.github.io/stable/generated/dirty_cat.MinHashEncoder.html>`_
- The `SuperVectorizer <https://dirty-cat.github.io/stable/generated/dirty_cat.SuperVectorizer.html>`_
- The `DatetimeEncoder <https://dirty-cat.github.io/stable/generated/dirty_cat.DatetimeEncoder.html>`_

You can find more details and examples on `the website <https://dirty-cat.github.io/>`_.

What dirty_cat does not
~~~~~~~~~~~~~~~~~~~~~~~

`Semantic similarities <https://en.wikipedia.org/wiki/Semantic_similarity>`_
are currently not supported.
For example, the similarity between *car* and *automobile* is outside the reach
of the methods implemented here.

This kind of problem is tackled by
`Natural Language Processing <https://en.wikipedia.org/wiki/Natural_language_processing>`_
methods.

dirty_cat can still help with handling typos and variations in this kind of setting.

Installation
------------

dirty_cat can be easily installed via `pip`::

    pip install dirty_cat

Dependencies
~~~~~~~~~~~~

Dependencies and minimal versions are listed in the file `min-requirements.txt <https://github.com/dirty-cat/dirty_cat/blob/master/requirements-min.txt>`_

Related projects
----------------

- `spark-dirty-cat <https://github.com/rakutentech/spark-dirty-cat>`_
  - a Scala implementation of dirty_cat for Spark ML
- `scikit-learn <https://github.com/scikit-learn/scikit-learn>`_
  - a very popular machine learning library; dirty_cat inherits its API
- `CleverCSV <https://github.com/alan-turing-institute/CleverCSV>`_
  - a package for dealing with dirty csv files
- `GAMA <https://github.com/openml-labs/gama>`_
  - a modular AutoML assistant that uses dirty_cat as part of its search space

Contributing
------------

If you want to encourage development of dirty_cat,
the best thing to do is to *spread the word*!

If you encounter an issue while using dirty_cat, please
`open an issue <https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue>`_ or
`submit a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_.
Don't hesitate, you're helping to make this project better for everyone!

Additional resources
--------------------

* `Introductory video (YouTube) <https://youtu.be/_GNaaeEI2tg>`_
* `Overview poster for EuroSciPy 2022 (Google Drive) <https://drive.google.com/file/d/1TtmJ3VjASy6rGlKe0txKacM-DdvJdIvB/view?usp=sharing>`_

References
----------

.. [1] Patricio Cerda, Gaël Varoquaux, Balázs Kégl. Similarity encoding for learning with dirty categorical variables. 2018. Machine Learning journal, Springer.
.. [2] Patricio Cerda, Gaël Varoquaux. Encoding high-cardinality string categorical variables. 2020. IEEE Transactions on Knowledge & Data Engineering.
