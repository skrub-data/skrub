
=================================================
skrub: Prepping tables for machine learning 
=================================================

.. toctree::
   :maxdepth: 2

.. currentmodule:: skrub

.. container:: larger-container

    `skrub` facilitates machine-learning with non-curated categories:
    **robust to morphological variants**, such as typos. See
    :ref:`examples <usage_examples>`, such as `the first one
    <https://skrub-data.github.io/stable/auto_examples/01_dirty_categories.html>`_,
    for an introduction to problems of dirty categories or misspelled
    entities.

|

.. raw:: html

    <div class="flex-container">
    <div class="flex-content">
    <span class="container-title">Automatic features from heterogeneous dataframes</span>

:class:`TableVectorizer`: a transformer to **easily turn a pandas
dataframe into a numpy array** suitable for machine learning -- a default
encoding pipeline you can tweak.

.. rst-class:: centered

    :ref:`An example <example_table_vectorizer>`

.. raw:: html

    </div>
    <div class="flex-content">
    <span class="container-title">OneHotEncoder but for non-normalized categories</span>


* :class:`GapEncoder`, scalable and interpretable, where each encoding
  dimension corresponds to a topic that summarizes substrings captured.
  :ref:`Example <example_gap_encoder>`

* :class:`SimilarityEncoder`, an enhanced one-hot encoder
  able to capture the string similarities in the data.
  :ref:`Example <example_similarity_encoder>`

* :class:`MinHashEncoder`, very scalable, suitable for big data.
  :ref:`Example <example_minhash_encoder>`

.. raw:: html

    </div>
    <div class="flex-content">
    <span class="container-title">Joining tables on non-normalized categories</span>

* :func:`fuzzy_join`, approximate matching using morphological similarity.
  :ref:`Example <example_fuzzy_join>`

* :class:`FeatureAugmenter`, a transformer for joining multiple tables together.
  :ref:`Example <example_feature_augmenter>`

.. raw:: html

    </div>
    <div class="flex-content">
    <span class="container-title">Deduplicating dirty categories</span>

:func:`deduplicate`, merging categories of similar morphology (spelling).

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
dirty-cat received funding from `project DirtyData
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
