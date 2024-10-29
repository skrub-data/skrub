.. currentmodule:: skrub

###
API
###

This page lists all available functions and classes of `skrub`.

.. _joining_dataframes_ref:

Joining dataframes
==================

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   Joiner
   AggJoiner
   MultiAggJoiner
   AggTarget
   InterpolationJoiner

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   fuzzy_join


.. _encoding_a_column_ref:

Encoding a column
=================

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   GapEncoder
   MinHashEncoder
   SimilarityEncoder
   DatetimeEncoder
   ToCategorical
   ToDatetime

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   to_datetime


.. _building_a_pipeline_ref:

Building a pipeline
===================

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   TableVectorizer
   SelectCols
   DropCols

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   tabular_learner


.. _generating_a_report_ref:

Generating an HTML report
=========================

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   TableReport

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   patch_display
   unpatch_display
   column_associations


.. _cleaning_a_dataframe_ref:

Cleaning a dataframe
====================

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   deduplicate


.. _downloading_a_dataset_ref:

.. currentmodule:: skrub.datasets

Downloading a dataset
=====================

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   fetch_employee_salaries
   fetch_medical_charge
   fetch_midwest_survey
   fetch_open_payments
   fetch_road_safety
   fetch_traffic_violations
   fetch_drug_directory
   fetch_world_bank_indicator
   fetch_movielens
   fetch_credit_fraud
   fetch_ken_table_aliases
   fetch_ken_types
   fetch_ken_embeddings
   make_deduplication_data
