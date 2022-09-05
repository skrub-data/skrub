dirty_cat
=========

.. image:: https://dirty-cat.github.io/stable/_static/dirty_cat.svg
   :align: center
   :alt: dirty_cat logo


|py_ver| |pypi_var| |pypi_dl| |codecov| |circleci| |Black|

.. |py_ver| image:: https://img.shields.io/pypi/pyversions/dirty_cat
.. |pypi_var| image:: https://img.shields.io/pypi/v/dirty_cat?color=informational
.. |pypi_dl| image:: https://img.shields.io/pypi/dm/dirty_cat
.. |codecov| image:: https://img.shields.io/codecov/c/github/dirty-cat/dirty_cat/master
.. |circleci| image:: https://img.shields.io/circleci/build/github/dirty-cat/dirty_cat/master?label=CircleCI
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   .. _Black: https://github.com/psf/black


dirty_cat is a Python module for machine-learning on dirty categorical variables.

Website: https://dirty-cat.github.io/

dirty_cat's SuperVectorizer automatically turns pandas data frames into
numerical arrays suitable for learning.

For a detailed description of the problem of encoding dirty categorical data,
see `Similarity encoding for learning with dirty categorical variables
<https://hal.inria.fr/hal-01806175>`_ [1]_ and `Encoding high-cardinality string categorical variables
<https://hal.inria.fr/hal-02171256v4>`_ [2]_.

Installation
------------

Dependencies
~~~~~~~~~~~~

dirty_cat requires:

- Python (>= 3.8)
- NumPy (>= 1.17.3)
- SciPy (>= 1.4.0)
- scikit-learn (>= 0.22.0)
- pandas (>= 1.2.0)

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of NumPy and SciPy,
the easiest way to install dirty_cat is using ``pip`` ::

    pip install -U --user dirty_cat

Other implementations
~~~~~~~~~~~~~~~~~~~~~~

-  Spark ML: https://github.com/rakutentech/spark-dirty-cat


References
~~~~~~~~~~

.. [1] Patricio Cerda, Gaël Varoquaux, Balázs Kégl. Similarity encoding for learning with dirty categorical variables. 2018. Machine Learning journal, Springer.
.. [2] Patricio Cerda, Gaël Varoquaux. Encoding high-cardinality string categorical variables. 2020. IEEE Transactions on Knowledge & Data Engineering.
