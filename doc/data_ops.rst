.. _user_guide_data_ops_index:

Complex multi-table pipelines with Data Ops
===========================================

Skrub provides an easy way to build complex, flexible machine learning pipelines.
There are several needs that are not easily addressed with standard scikit-learn
tools such as :class:`~sklearn.pipeline.Pipeline` and
:class:`~sklearn.compose.ColumnTransformer`, and for which the Skrub DataOps offer
a solution.

A high-level overview of Data Ops is provided in :ref:`getting_started_with_data_ops`.
More detail is available in the other pages in this section, where we cover all
about the skrub Data Ops, from starting out with a
simple example, to more advanced concepts like parameter tuning and and pipeline
validation.

Data Ops basic concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 3

   modules/data_ops/basics/what_are_data_ops
   modules/data_ops/basics/building_data_ops_plan
   modules/data_ops/basics/using_previews
   modules/data_ops/basics/direct_access_methods
   modules/data_ops/basics/control_flow
   modules/data_ops/basics/data_ops_vs_alternatives

Building a complex pipeline with the skrub Data Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   modules/data_ops/ml_pipeline/applying_ml_estimators
   modules/data_ops/ml_pipeline/applying_different_transformers
   modules/data_ops/ml_pipeline/documenting_data_ops_plan
   modules/data_ops/ml_pipeline/evaluating_debugging_data_ops
   modules/data_ops/ml_pipeline/using_part_of_data_ops_plan
   modules/data_ops/ml_pipeline/subsampling_data

Tuning and validating Skrub DataOps plans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   modules/data_ops/validation/tuning_validating_data_ops
   modules/data_ops/validation/hyperparameter_tuning
   modules/data_ops/validation/nested_cross_validation
   modules/data_ops/validation/nesting_choices_choosing_pipelines
   modules/data_ops/validation/exporting_data_ops
