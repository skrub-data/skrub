dirty_cat
=========

dirty_cat is a Python module for machine-learning on dirty categorical variables.

Website: https://dirty-cat.github.io/

It implements the following encoders for categorical variables:

- Similarity encoding [1]_
- Target encoding [2]_

Installation
------------

Dependencies
~~~~~~~~~~~~

dirty_cat requires:

- python (>= 3.5)
- numpy (>= 1.8.2)
- scipy (>= 0.13.3)
- scikit-learn (>= 0.19.0)

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


.. [2] Patricio Cerda, Gaël Varoquaux, Balázs Kégl. Similarity encoding for learning with dirty categorical variables. 2018.
