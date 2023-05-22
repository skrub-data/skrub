User guide
===========

Encoding dirty categories
--------------------------

For a detailed description of the problem of encoding dirty categorical data,
see `Similarity encoding for learning with dirty categorical variables
<https://hal.inria.fr/hal-01806175>`_ [1]_ and `Encoding high-cardinality
string categorical variables <https://hal.inria.fr/hal-02171256v4>`_ [2]_.



About
------

skrub is a young project born from research. We really need people
giving feedback on successes and failures with the different techniques on real
world data, and pointing us to open datasets on which we can do more
empirical work.
skrub received funding from `project DirtyData
<https://project.inria.fr/dirtydata/>`_ (ANR-17-CE23-0018).

.. [1] Patricio Cerda, Gaël Varoquaux. Encoding high-cardinality string categorical variables. 2020. IEEE Transactions on Knowledge & Data Engineering.
.. [2] Patricio Cerda, Gaël Varoquaux, Balázs Kégl. Similarity encoding for learning with dirty categorical variables. 2018. Machine Learning journal, Springer.


Related projects
-----------------

- `scikit-learn <https://github.com/scikit-learn/scikit-learn>`_
  - a very popular machine learning library; *skrub* inherits its API
- `categorical-encoding <https://contrib.scikit-learn.org/category_encoders/>`_
  - scikit-learn compatible classic categorical encoding schemes
- `spark-dirty-cat <https://github.com/rakutentech/spark-dirty-cat>`_
  - a Scala implementation of skrub for Spark ML
- `CleverCSV <https://github.com/alan-turing-institute/CleverCSV>`_
  - a package for dealing with dirty csv files
- `GAMA <https://github.com/openml-labs/gama>`_
  - a modular AutoML assistant that uses *skrub* as part of its search space

