User guide
===========

Skrub facilitates preparing tables for machine learning. It is not a
replacement for tools such as SQL or pandas, as these tools are much more
versatile. Rather it gives higher-level operations for machine-learning,
typically with `scikit-learn <http://scikit-learn.org>`_ with its
`pipelines <https://scikit-learn.org/stable/modules/compose.html>`_.

.. topic:: Skrub highlights:

 - facilitates separating the train and test operations, for model
   selection and to put models in production

 - enables statistical and imperfect assembly, as machine-learning models
   can typically retrieve signals even in noisy data.

|

.. include:: includes/big_toc_css.rst

.. toctree::

   encoding
   assembling
   cleaning
   development
