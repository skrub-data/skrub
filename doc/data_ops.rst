.. _user_guide_data_ops_index:

Complex multi-table pipelines with Data Ops
===========================================

Skrub provides an easy way to build complex, flexible machine learning pipelines.
There are several needs that are not easily addressed with standard scikit-learn
tools such as :class:`~sklearn.pipeline.Pipeline` and
:class:`~sklearn.compose.ColumnTransformer`, and for which the Skrub DataOps offer
a solution:

- Multiple tables: We often have several tables of different shapes (for
  example, "Customers", "Orders", and "Products" tables) that need to be
  processed and assembled into a design matrix ``X``. The target ``y`` may also
  be the result of some data processing. Standard scikit-learn estimators do not
  support this, as they expect right away a single design matrix ``X`` and a
  target array ``y``, with one row per observation.
- DataFrame wrangling: Performing typical DataFrame operations such as
  projections, joins, and aggregations should be possible and allow leveraging
  the powerful and familiar APIs of `Pandas <https://pandas.pydata.org>`_ or
  `Polars <https://docs.pola.rs/>`_.
- Hyperparameter tuning: Choices of estimators, hyperparameters, and even
  the pipeline architecture can be guided by validation scores. Specifying
  ranges of possible values outside of the pipeline itself (as in
  :class:`~sklearn.model_selection.GridSearchCV`) is difficult in complex
  pipelines.
- Iterative development: Building a pipeline step by step while inspecting
  intermediate results allows for a short feedback loop and early discovery of
  errors.

In this section we cover all about the skrub Data Ops, from starting out with a
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
