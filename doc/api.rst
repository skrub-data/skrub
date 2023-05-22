
API
=================

This page lists all the functions and classes of skrub:

.. currentmodule:: skrub


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
