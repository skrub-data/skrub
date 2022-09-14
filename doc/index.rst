
===============================================
dirty_cat: machine learning on dirty categories
===============================================

.. toctree::
   :maxdepth: 2

.. currentmodule:: dirty_cat

.. container:: larger-container

    `dirty_cat` facilitates machine-learning on non-curated categories:
    **robust to morphological variants**, such as typos.

|

.. raw:: html

    <div class="flex-container">
    <div class="flex-content">
    <span class="container-title">Automatic features from heterogeneous dataframes</span>

:class:`SuperVectorizer`: a transformer **automatically turning a pandas
dataframe into a numpy array** for machine learning -- a default encoding
pipeline you can tweak.

.. rst-class:: centered

    :ref:`An example <example_super_vectorizer>`

.. raw:: html

    </div>
    <div class="flex-content">
    <span class="container-title">OneHotEncoder but for non-normalized categories</span>


* :class:`GapEncoder`, scalable and interpretable, where each encoding
  dimension corresponds to a topic that summarizes substrings captured.

* :class:`SimilarityEncoder`, a simple modification of one-hot encoding
  to capture the strings.

* :class:`MinHashEncoder`, very scalable

.. raw:: html

    </div>
    </div>

.. container:: right-align

   `Recent changes <CHANGES.html>`_

   `Contributing <development.html>`_

.. container:: install_instructions

    :Installing: `$ pip install --user --upgrade dirty_cat`



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


Usage examples
===============

.. container:: larger-container

  .. include:: auto_examples/index.rst
    :start-line: 2
    :end-before: .. rst-class:: sphx-glr-signature

|

For a detailed description of the problem of encoding dirty categorical data,
see `Similarity encoding for learning with dirty categorical variables
<https://hal.inria.fr/hal-01806175>`_ [1]_ and `Encoding high-cardinality
string categorical variables <https://hal.inria.fr/hal-02171256v4>`_ [2]_.


API documentation
=================

Vectorizing a dataframe
------------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   SuperVectorizer

Dirty Category encoders
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
-----------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   DatetimeEncoder

Data download
-----------------------

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
   datasets.get_data_dir

About
=========

dirty_cat is for now a repository for ideas coming out of a research
project: there is still little known about the problems of dirty
categories. Tradeoffs will emerge in the long run. We really need people
giving feedback on success and failures with the different techniques and
pointing us to open datasets on which we can do more empirical work.
dirty-cat received funding from `project DirtyData
<https://project.inria.fr/dirtydata/>`_ (ANR-17-CE23-0018).

.. [1] Patricio Cerda, Gaël Varoquaux. Encoding high-cardinality string categorical variables. 2020. IEEE Transactions on Knowledge & Data Engineering.
.. [2] Patricio Cerda, Gaël Varoquaux, Balázs Kégl. Similarity encoding for learning with dirty categorical variables. 2018. Machine Learning journal, Springer.


.. seealso::

   Many classic categorical encoding schemes are available here:
   https://contrib.scikit-learn.org/category_encoders/

   Similarity encoding in also available in Spark ML:
   https://github.com/rakutentech/spark-dirty-cat

