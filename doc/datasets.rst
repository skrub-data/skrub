.. _datasets:


.. only:: indpage
   ================
   Example datasets
   ================

.. currentmodule:: skrub

skrub provides fetching methods for datasets, 
which are generally used in examples and benchmarks.

It consists of generated, embeddings and real world data.

Real world datasets
-------------------

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:
   :caption: Real world datasets

   datasets.fetch_employee_salaries
   datasets.fetch_medical_charge
   datasets.fetch_midwest_survey
   datasets.fetch_open_payments
   datasets.fetch_road_safety
   datasets.fetch_traffic_violations
   datasets.fetch_drug_directory
   datasets.fetch_world_bank_indicator
   datasets.fetch_figshare

Generated dataset
-----------------

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:
   :caption: Generated dataset

   datasets.make_deduplication_data

Embeddings
----------

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:
   :caption: Embeddings

   datasets.fetch_ken_embeddings
   datasets.fetch_ken_types
   datasets.fetch_ken_table_aliases
