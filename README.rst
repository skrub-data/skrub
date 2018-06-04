dirty_cat
=========

dirty_cat is a Python module for machine-learning on dirty categorical variables.

Website: https://dirty-cat.github.io/

It implements the following encoders for categorical variables:

- Similarity encoding [CeVa18]_
- Target encoding [MiBa01]_

Installation
------------

Dependencies
~~~~~~~~~~~~

dirty_cat requires:

- Python (>= 3.5)
- NumPy (>= 1.8.2)
- SciPy (>= 0.13.3)
- scikit-learn

Optional dependency:

- python-Levenshtein for faster edit distances (not used for the n-gram
  distance)

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of NumPy and SciPy,
the easiest way to install dirty_cat is using ``pip`` ::

    pip install -U ...


References
~~~~~~~~~~

.. [1] Micci-Barreca, D.: A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. 2001. ACM SIGKDD Explorations Newsletter, 3(1), 27-32.


.. [2] Patricio Cerda, Gaël Varoquaux, Balázs Kégl. Similarity encoding for learning with dirty categorical variables. 2018. https://hal.inria.fr/hal-01806175
