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
   StringEncoder

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   to_datetime

Deep Learning
-------------

These encoders require installing additional dependencies around torch.
See the "deep learning dependencies" section in the :ref:`installation_instructions`
guide for more details.

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   TextEncoder


.. _building_a_pipeline_ref:

Building a pipeline
===================

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   TableVectorizer
   Cleaner
   SelectCols
   DropCols

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   tabular_learner


.. _expressions_ref:

Skrub Expressions
=================

The ``tabular_learner`` provides a pre-defined, default pipeline for datasets that contain a simple table.
For more control or in order to build pipelines for more datasets, use the skrub expressions.


.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   as_expr
   choose_bool
   choose_float
   choose_from
   choose_int
   cross_validate
   deferred
   eval_mode
   optional
   train_test_split
   var
   X
   y

.. autosummary::
   :toctree: generated/
   :template: expr_class.rst
   :nosignatures:

   Expr

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst
   :nosignatures:

   Expr.skb.apply
   Expr.skb.clone
   Expr.skb.concat_horizontal
   Expr.skb.cross_validate
   Expr.skb.describe_param_grid
   Expr.skb.describe_steps
   Expr.skb.draw_graph
   Expr.skb.drop
   Expr.skb.eval
   Expr.skb.freeze_after_fit
   Expr.skb.full_report
   Expr.skb.get_data
   Expr.skb.get_estimator
   Expr.skb.get_grid_search
   Expr.skb.get_randomized_search
   Expr.skb.if_else
   Expr.skb.mark_as_X
   Expr.skb.mark_as_y
   Expr.skb.match
   Expr.skb.select
   Expr.skb.set_description
   Expr.skb.set_name
   Expr.skb.train_test_split

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst
   :nosignatures:

   Expr.skb.description
   Expr.skb.is_X
   Expr.skb.is_y
   Expr.skb.name
   Expr.skb.applied_estimator

.. autosummary::
   :toctree: generated/
   :template: base.rst
   :nosignatures:

   ExprEstimator
   ParamSearch

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

   fetch_bike_sharing
   fetch_country_happiness
   fetch_credit_fraud
   fetch_drug_directory
   fetch_employee_salaries
   fetch_flight_delays
   fetch_ken_embeddings
   fetch_ken_table_aliases
   fetch_ken_types
   fetch_medical_charge
   fetch_midwest_survey
   fetch_movielens
   fetch_open_payments
   fetch_toxicity
   fetch_traffic_violations
   fetch_videogame_sales
   get_data_dir
   make_deduplication_data
