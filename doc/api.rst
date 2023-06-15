API
===

This page lists all available functions and classes of skrub

.. currentmodule:: skrub

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
   datasets.fetch_ken_table_aliases
   datasets.fetch_ken_types
   datasets.fetch_ken_embeddings
   datasets.fetch_data_dir
   datasets.make_deduplication_data
