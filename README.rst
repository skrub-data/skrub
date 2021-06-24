dirty_cat
=========

dirty_cat is a Python module for machine-learning on dirty categorical variables.

Website: https://dirty-cat.github.io/

For a detailed description of the problem of encoding dirty categorical data,
see `Similarity encoding for learning with dirty categorical variables
<https://hal.inria.fr/hal-01806175>`_ [1]_ and `Encoding high-cardinality string categorical variables
<https://hal.inria.fr/hal-02171256v4>`_ [2]_.

Installation
------------

Dependencies
~~~~~~~~~~~~

dirty_cat requires:

- Python (>= 3.6)
- NumPy (>= 1.16)
- SciPy (>= 1.2)
- scikit-learn (>= 0.20.0)

Optional dependencies:

- python-Levenshtein for faster edit distances (not used for the n-gram
  distance)
- fasttext (>= 0.9.2) to allow encoding of categorical data with pretrained
  fastText models with the class `PretrainedFastText`

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
