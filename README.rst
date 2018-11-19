dirty_cat
=========

dirty_cat is a Python module for machine-learning on dirty categorical variables.

Website: https://dirty-cat.github.io/

For a detailed description of the problem of encoding dirty categorical data,
see `Similarity encoding for learning with dirty categorical variables
<https://hal.inria.fr/hal-01806175>`_ [1]_.

Installation
------------

Dependencies
~~~~~~~~~~~~

dirty_cat requires:

- Python (>= 3.5)
- NumPy (>= 1.8.2)
- SciPy (>= 1.0.1)
- scikit-learn (>= 0.20.0)

Optional dependency:

- python-Levenshtein for faster edit distances (not used for the n-gram
  distance)

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

.. [1] Patricio Cerda, Gaël Varoquaux, Balázs Kégl. Similarity encoding for learning with dirty categorical variables. 2018, Machine Learning journal, Springer.
