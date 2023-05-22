
============================================
skrub: Prepping tables for machine learning 
============================================

.. toctree::
   :maxdepth: 2

.. currentmodule:: skrub

.. container:: larger-container

    - Built for scikit-learn.
    - Robust to dirty data.
    - Easy learning on pandas dataframes.

|

.. raw:: html

    <div class="flex-container">
    <div class="flex-content">
    <span class="container-title">Assembling
    </span>

* :func:`fuzzy_join`, Joining tables on non-normalized categories with
  approximate matching. :ref:`Example <example_fuzzy_join>`

* :class:`FeatureAugmenter`, a transformer for joining multiple tables together.
  :ref:`Example <example_feature_augmenter>`

.. raw:: html

    </div>
    <div class="flex-content">
    <span class="container-title">Encoding</span>

Feature matrices from dataframes:

* :class:`TableVectorizer`: **easily turn a pandas
  dataframe into a numpy array** suitable for machine learning
  :ref:`An example <example_table_vectorizer>`

* :class:`GapEncoder`, OneHotEncoder but robust to typos or
   non-normalized categories

  :ref:`Example <example_gap_encoder>`

.. raw:: html

    </div>
    <div class="flex-content">
    <span class="container-title">Cleaning</span>

Deduplication :func:`deduplicate`, merging categories of similar
morphology (spelling).

.. rst-class:: centered

     :ref:`An example <example_deduplication>`

.. raw:: html

    </div>
    </div>

.. container:: right-align

   `Recent changes <CHANGES.html>`_

   `Contributing <development.html>`_

.. container:: install_instructions

    :Installing: ``$ pip install --user --upgrade skrub``

.. _usage_examples:

Usage examples
==============

.. container:: larger-container

  .. include:: auto_examples/index.rst
    :start-line: 5
    :end-before: .. rst-class:: sphx-glr-signature

|

.. raw:: html

    <div class="video" style="border: 0px;">
    <iframe style="display: block; margin: auto; width: 100%;" width="560" height="315"
     src="https://www.youtube.com/embed/_GNaaeEI2tg" frameborder="0"
     allow="accelerometer; autoplay; clipboard-write;
     encrypted-media; gyroscope; picture-in-picture" allowfullscreen
    ></iframe></div>
    <div class="flex-content" style="border: 0px;">

.. raw:: html

    </div>


For a detailed description of the problem of encoding dirty categorical data,
see `Similarity encoding for learning with dirty categorical variables
<https://hal.inria.fr/hal-01806175>`_ [1]_ and `Encoding high-cardinality
string categorical variables <https://hal.inria.fr/hal-02171256v4>`_ [2]_.


API documentation
=================

Vectorizing a dataframe
-----------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   TableVectorizer

Dirty category encoders
-----------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   GapEncoder
   MinHashEncoder
   SimilarityEncoder
   TargetEncoder

Other encoders
--------------

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   DatetimeEncoder

Joining tables
--------------

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   fuzzy_join

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   FeatureAugmenter

Deduplication: merging variants of the same entry
-------------------------------------------------

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   deduplicate


Data download and generation
----------------------------

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:

   datasets.fetch_employee_salaries
   datasets.fetch_medical_charge
   datasets.fetch_midwest_survey
   datasets.fetch_open_payments
   datasets.fetch_road_safety
   datasets.fetch_traffic_violations
   datasets.fetch_drug_directory
   datasets.fetch_world_bank_indicator
   datasets.get_ken_table_aliases
   datasets.get_ken_types
   datasets.get_ken_embeddings
   datasets.get_data_dir
   datasets.make_deduplication_data

About
=====

skrub is a young project born from research. We really need people
giving feedback on successes and failures with the different techniques on real
world data, and pointing us to open datasets on which we can do more
empirical work.
skrub received funding from `project DirtyData
<https://project.inria.fr/dirtydata/>`_ (ANR-17-CE23-0018).

.. [1] Patricio Cerda, Gaël Varoquaux. Encoding high-cardinality string categorical variables. 2020. IEEE Transactions on Knowledge & Data Engineering.
.. [2] Patricio Cerda, Gaël Varoquaux, Balázs Kégl. Similarity encoding for learning with dirty categorical variables. 2018. Machine Learning journal, Springer.


Related projects
================

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
