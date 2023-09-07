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
   AggJoiner
   AggTarget


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

   <h2>Dataframes operations</h2>

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:
   :caption: DataFrames operations

   dataframe.get_df_namespace

.. raw:: html

   <h3>Pandas</h3>

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:
   :caption: Pandas operations

   dataframe.is_pandas
   dataframe.pd_aggregate
   dataframe.pd_join

.. raw:: html

   <h3>Polars</h3>

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:
   :caption: Polars operations

   dataframe.is_polars
   dataframe.pl_aggregate
   dataframe.pl_join

.. raw:: html

   <h2>Data download and generation</h2>

.. autosummary::
   :toctree: generated/
   :template: function.rst
   :nosignatures:
   :caption: Data download and generation

   datasets.fetch_employee_salaries
   datasets.fetch_medical_charge
   datasets.fetch_midwest_survey
   datasets.fetch_open_payments
   datasets.fetch_road_safety
   datasets.fetch_traffic_violations
   datasets.fetch_drug_directory
   datasets.fetch_world_bank_indicator
   datasets.fetch_movielens
   datasets.fetch_ken_table_aliases
   datasets.fetch_ken_types
   datasets.fetch_ken_embeddings
   datasets.make_deduplication_data
