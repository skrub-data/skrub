.. _user_guide_data_ops_index:

.. currentmodule:: skrub

Building complete pipelines with DataOps
========================================

A skrub DataOp is a complete machine learning pipeline —from data loading and
wrangling to the final prediction— in a single object that can be fitted, tuned,
cross-validated, and saved in a file like any scikit-learn estimator.

By integrating the whole data processing, DataOps help to validate pipelines
while **avoiding data leakage**, to **tune complex modelling choices**, and to keep
track of important **fitted (learned) state**.

To solve a machine-learning task we often need to combine multiple operations
such as loading and filtering data, joining tables and computing aggregations,
extracting numerical features, and fitting a classifier or regressor.

**Storing state**  Each of those operations may need to be fitted: to learn some
information from training data and reuse it to apply consistent transformations
to new data. This is the case for transformers like the
:class:`~sklearn.preprocessing.StandardScaler` and :class:`TableVectorizer` and
estimators like :class:`~sklearn.ensemble.RandomForestClassifier`.

**Tuning**  Moreover, each processing step may involve decisions that need to be
tuned (*tuning* means finding the value that gives the best predictive
performance), for example: what weather forecast features should I include to
predict the load on an electric grid? How should I encode a product description
to help predict the product's category? What learning rate to set on a
:class:`~sklearn.ensemble.HistGradientBoostingRegressor`?

**Validation**  Finally, the quality of predictions must be evaluated on
held-out data (with a train/test split or cross-validation), taking care to
**avoid leakage** of test data into the training set.

Separating the data wrangling from the fitted estimator prevents correctly
handling the tasks above. Skrub DataOps help by binding an arbitrary set of
transformations of any number of inputs in a single estimator. These
transformations can be easily parametrized with tunable choices. The resulting
objects have built-in methods for cross-validation and tuning with either Optuna
or scikit-learn, and for inspecting runs and intermediate results. Once fitted,
they can be saved in a file, loaded, applied to new data as easily as a single
:class:`~sklearn.linear_model.LogisticRegression`.

.. dropdown:: Going beyond the scikit-learn Pipeline
  :color: primary

  To some extent, the DataOps exist for the same reasons as the simpler
  scikit-learn :class:`sklearn.pipeline.Pipeline` used in other parts of this
  documentation. However the Pipeline is too limited for many real-world problems:
  it can only represent a linear sequence of scikit-learn transformers, the design
  matrix and target variables must be constructed and divided into training and
  testing sets outside of the pipeline and the number of rows cannot change, only
  a single table can be handled, hyperparameter choices are difficult to define,
  etc. . Skrub DataOps remove those limitations and add several useful features
  such as interactive previews and integration with Optuna.

A quick overview of DataOps
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial walks through the main components of the DataOps on a simple
example:

.. toctree::
   :maxdepth: 2

   auto_tutorials/1111_data_ops_quick_tour


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
