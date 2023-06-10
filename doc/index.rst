
.. raw:: html

    <div class="container-fluid sk-landing-bg">
      <div class="container sk-landing-container">
        <div class="row">
          <div class="col-md-6 mb-3 mb-md-0">
            <h1 class="sk-landing-header text-white text-monospace">skrub</h1>
            <h4 class="sk-landing-subheader text-white font-italic mb-3">Prepping tables for machine learning</h4>
          </div>
          <div class="col-md-6 d-flex">
            <ul class="sk-landing-header-body">
              <li>Built for scikit-learn</li>
              <li>Robust to dirty data</li>
              <li>Easy learning on pandas dataframes</li>
            </ul>
          </div>
        </div>
      </div>
    </div>


.. toctree::
   :maxdepth: 2

.. currentmodule:: skrub


|

.. image:: _static/skrub_pipeline.svg

|

.. raw:: html

    <div class="container-fluid">
    <div class="row">
    <div class="col-lg-4">
    <div class="sd-card sd-shadow-sm">
    <div class="card-body">
    <h4 class="card-title">Assembling</h4>

:func:`fuzzy_join`, Joining tables on non-normalized categories with approximate matching. :ref:`Example <example_fuzzy_join>`

:class:`FeatureAugmenter`, a transformer for joining multiple tables together. :ref:`Example <example_feature_augmenter>`

.. raw:: html

    </div>
    </div>
    </div>
    <div class="col-lg-4">
    <div class="sd-card sd-shadow-sm">
    <div class="card-body">
    <h4 class="card-title">Encoding</h4>

:class:`TableVectorizer`: **turn a pandas dataframe into a numerical array** for machine learning. :ref:`Example <example_table_vectorizer>`

:class:`GapEncoder`, OneHotEncoder but robust to typos or non-normalized categories. :ref:`Example <example_gap_encoder>`

.. raw:: html

    </div>
    </div>
    </div>
    <div class="col-lg-4">
    <div class="sd-card sd-shadow-sm">
    <div class="card-body">
    <h4 class="card-title">Cleaning</h4>

:func:`deduplicate`: merge categories of similar morphology (spelling). :ref:`Example <example_deduplication>`

.. raw:: html

    </div>
    </div>
    </div>
    </div>
    </div>


|

.. container:: right-align

   `Recent changes <CHANGES.html>`_

   `Contributing <CONTRIBUTING.html>`_

.. toctree::
   :hidden:

   documentation
   api
   auto_examples/index
