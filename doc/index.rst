
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

.. toctree::
   :hidden:

   documentation
   api
   auto_examples/index


