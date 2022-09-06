dirty_cat
=========

.. image:: https://dirty-cat.github.io/stable/_static/dirty_cat.svg
   :align: center
   :alt: dirty_cat logo


|py_ver| |pypi_var| |pypi_dl| |codecov| |circleci|

.. |py_ver| image:: https://img.shields.io/pypi/pyversions/dirty_cat
.. |pypi_var| image:: https://img.shields.io/pypi/v/dirty_cat?color=informational
.. |pypi_dl| image:: https://img.shields.io/pypi/dm/dirty_cat
.. |codecov| image:: https://img.shields.io/codecov/c/github/dirty-cat/dirty_cat/master
.. |circleci| image:: https://img.shields.io/circleci/build/github/dirty-cat/dirty_cat/master?label=CircleCI

`dirty_cat <https://dirty-cat.github.io/>`_ is a Python library
that facilitates machine-learning on dirty categorical variables.

The techniques implemented are based on science!
For a detailed description of the problem of encoding dirty categorical data, see
`Similarity encoding for learning with dirty categorical variables <https://hal.inria.fr/hal-01806175>`_ [1]_
and `Encoding high-cardinality string categorical variables <https://hal.inria.fr/hal-02171256v4>`_ [2]_.

If you like the package, please *spread the word*, and ⭐ the repository!

When to use?
------------

dirty_cat assumes a correlation between the textual (morphological) similarities
in the feature(s) and the similarities in the target.

For this example, let's consider the species

dirty_cat encoders can be easily integrated in already existing
scikit-learn pipelines as they are compliant with the library's principles.

As such, they can be drop-in replacements for other encoders !

Encoders
~~~~~~~~

dirty_cat provides various tools for dealing with dirty data:

- The `SimilarityEncoder <https://dirty-cat.github.io/stable/generated/dirty_cat.SimilarityEncoder.html>`_
- The `GapEncoder <https://dirty-cat.github.io/stable/generated/dirty_cat.GapEncoder.html>`_
- The `MinHashEncoder <https://dirty-cat.github.io/stable/generated/dirty_cat.MinHashEncoder.html>`_
- The `SuperVectorizer <https://dirty-cat.github.io/stable/generated/dirty_cat.SuperVectorizer.html>`_
- The `DatetimeEncoder <https://dirty-cat.github.io/stable/generated/dirty_cat.DatetimeEncoder.html>`_

You can find more details and examples on `the website <https://dirty-cat.github.io/>`_.

Benchmark
~~~~~~~~~



Installation
------------

dirty_cat can be easily installed via `pip`::

    pip install dirty_cat

Dependencies
~~~~~~~~~~~~

Dependencies and minimal versions are listed in the file `min-requirements.txt <https://github.com/dirty-cat/dirty_cat/blob/master/requirements-min.txt>`_

Other implementations
---------------------

-  Spark ML: https://github.com/rakutentech/spark-dirty-cat

Contributing
------------

If you want to encourage development of dirty_cat,
the best thing to do now is to *spread the word*!

If you encounter an issue while using dirty_cat, please
`open an issue <https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue>`_ or
`submit a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_.
Don't hesitate, you're helping to make this project better for everyone!

References
----------

.. [1] Patricio Cerda, Gaël Varoquaux, Balázs Kégl. Similarity encoding for learning with dirty categorical variables. 2018. Machine Learning journal, Springer.
.. [2] Patricio Cerda, Gaël Varoquaux. Encoding high-cardinality string categorical variables. 2020. IEEE Transactions on Knowledge & Data Engineering.

Citations
~~~~~~~~~

If you're using dirty_cat in a scientific publication, we would greatly appreciate
citations:

Notes
~~~~~

The library is provided under the `BSD 3-clause licence <https://github.com/dirty-cat/dirty_cat/blob/master/LICENSE.txt>`_.
