.. _user_guide_data_ops_index:

.. currentmodule:: skrub

Building complete pipelines with DataOps
========================================

A skrub DataOp is a complete machine learning pipeline, from data loading and
wrangling to the final prediction, in a single object that can be fitted, tuned
and cross-validated, and saved in a file like any scikit-learn estimator.

To solve a machine-learning task we often need to combine multiple operations
such as loading and filtering data, joining tables and computing aggregations,
extracting numerical features, and fitting a classifier or regressor.

**Storing state:** each of those operations may need to be fitted, to learn some
information from training data and reuse it to apply consistent transformations
to new data. This is the case for transformers like the
:class:`~sklearn.preprocessing.StandardScaler` and :class:`TableVectorizer` and
estimators like :class:`~sklearn.ensemble.RandomForestClassifier`.

**Tuning:** moreover, each processing step may involve decisions that need to be
tuned (tuning means finding the value that gives the best predictive
performance), for example: what weather forecast data should I join to predict
the load on an electric grid? How should I encode the product description to
predict its category? What learning rate to set on a
:class:`~sklearn.ensemble.HistGradientBoostingRegressor`?

**Validation:** finally, the quality of predictions must be evaluated on
held-out data (with a train/test split or cross-validation), taking care to
avoid leakage of information about the test set into the training set.

Skrub DataOps help by binding an arbitrary set of transformations of any number of
inputs in a single estimator. These transformations can be parametrized with
values to be tuned. The resulting objects have built-in methods for
cross-validation and tuning with either Optuna or scikit-learn, and for
inspecting runs and intermediate results. Once fitted, they can be saved in a
file, loaded, applied to new data as easily as a single
:class:`~sklearn.linear_model.LogisticRegression`.

To some extent, the DataOps exist for the same reason as the simpler
scikit-learn :class:`sklearn.pipeline.Pipeline` used in many examples of this
documentation. However the Pipeline is too limited for many real-world problems:
it can only represent a linear sequence of scikit-learn transformers, the design
matrix and target variables must be constructed and divided into training and
testing sets outside of the pipeline and the number of rows cannot change, only
a single table can be handled, hyperparameter choices are difficult to define,
etc. . Skrub DataOps remove those limitations and add several useful features
such as interactive previews and integration with Optuna.

Data Ops basic concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 3

   modules/data_ops/basics/what_are_data_ops
   modules/data_ops/basics/building_data_ops_plan
   auto_tutorials/1110_data_ops_intro
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

Tuning and validating skrub DataOps plans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   modules/data_ops/validation/tuning_validating_data_ops
   modules/data_ops/validation/hyperparameter_tuning
   modules/data_ops/validation/nested_cross_validation
   modules/data_ops/validation/nesting_choices_choosing_pipelines
   modules/data_ops/validation/exporting_data_ops
   modules/data_ops/validation/tuning_with_optuna
