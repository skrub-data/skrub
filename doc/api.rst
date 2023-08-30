#############
API reference
#############

.. raw:: html

  <style type="text/css">
  article section h2 {
    margin-top: 4ex;
  }
  </style>

This page lists all available functions and classes of `skrub`.

.. currentmodule:: skrub

.. raw:: html

   <h2>Joining tables</h2>

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:
   :caption: Joining tables

   fuzzy_join

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:

   Joiner

.. raw:: html

   <h2>Vectorizing a dataframe</h2>

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:
   :caption: Vectorizing a dataframe

   TableVectorizer

.. raw:: html

   <h2>Dirty category encoders</h2>

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:
   :caption: Dirty category encoders

   GapEncoder
   MinHashEncoder
   SimilarityEncoder
   TargetEncoder

.. raw:: html

   <h2>Other encoders</h2>

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :nosignatures:
   :caption: Other encoders

   DatetimeEncoder

.. raw:: html

   <h2>Deduplication: merging variants of the same entry</h2>

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:
   :caption: Deduplication: merging variants of the same entry

   deduplicate

.. raw:: html

   <h2>Data download and generation</h2>

.. toctree::
   :maxdepth: 2
   :caption: Data download and generation

   datasets
