.. _getting_started_with_data_ops:

.. currentmodule:: skrub

.. |var| replace:: :func:`~skrub.var`
.. |X| replace:: :func:`~skrub.X`
.. |y| replace:: :func:`~skrub.y`
.. |choose_float| replace:: :func:`~skrub.choose_float`
.. |choose_int| replace:: :func:`~skrub.choose_int`
.. |choose_from| replace:: :func:`~skrub.choose_from`
.. |optional| replace:: :func:`~skrub.optional`
.. |deferred| replace:: :func:`~skrub.deferred`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |MinHashEncoder| replace:: :class:`~skrub.MinHashEncoder`
.. |StringEncoder| replace:: :class:`~skrub.StringEncoder`
.. |skb.apply| replace:: :meth:`~skrub.DataOp.skb.apply`
.. |skb.mark_as_X| replace:: :meth:`~skrub.DataOp.skb.mark_as_X`
.. |skb.mark_as_y| replace:: :meth:`~skrub.DataOp.skb.mark_as_y`
.. |skb.make_learner| replace:: :meth:`~skrub.DataOp.skb.make_learner`
.. |skb.make_randomized_search| replace:: :meth:`~skrub.DataOp.skb.make_randomized_search`
.. |skb.make_grid_search| replace:: :meth:`~skrub.DataOp.skb.make_grid_search`
.. |skb.cross_validate| replace:: :meth:`~skrub.DataOp.skb.cross_validate`
.. |skb.full_report| replace:: :meth:`~skrub.DataOp.skb.full_report`

Data Ops crash course
=============================

What are Data Ops?
-----------------

**Data Ops** provide a powerful framework for building complex, flexible machine
learning pipelines with tabular data. They allow you to express data transformations
and model training as a directed acyclic graph (DAG) of operations that can be easily
replayed, tuned, and deployed. Data Ops solve several key challenges in machine
learning workflows that traditional scikit-learn tools don't fully address, such as
working with multiple tables, incorporating DataFrame APIs, and integrating
hyperparameter tuning directly within your pipeline definition.

For a comprehensive introduction to Data Ops concepts, see :ref:`user_guide_data_ops_intro` and
the introductory example :ref:`sphx_glr_auto_examples_data_ops_11_data_ops_intro.py`.

Key Features of Data Ops
------------------------

**Multi-Table Processing**
  Data Ops enable working with multiple interconnected tables of different shapes
  simultaneously. Instead of forcing early merging of tables into a single design
  matrix, you can process tables individually and join them at the optimal point
  in your pipeline. This is particularly valuable when working with relational
  data like products and orders tables, where you might need to vectorize products,
  aggregate them at the order level, and then merge with order metadata before
  training a classifier - all within a single cohesive pipeline that handles proper
  train/test splitting. For a practical example, see :ref:`sphx_glr_auto_examples_data_ops_12_multiple_tables.py`.

**Seamless DataFrame Integration**
  Work directly with the Pandas or Polars DataFrame APIs without leaving
  the pipeline structure. Data Ops seamlessly track all operations applied to
  DataFrames, including selections, projections, joins, and aggregations, recording
  them in the computational graph for later replay on new data. Learn more in the user guide
  section :ref:`user_guide_data_ops_plan` on building Data Ops plans.

**Integrated Hyperparameter Tuning**
  Define hyperparameter ranges directly within your pipeline using ``choose_*``
  functions like |choose_float|, |choose_int|, and |choose_from|. Instead of
  defining a separate parameter grid outside your pipeline (like with scikit-learn's
  GridSearchCV), you simply replace concrete values with choice objects right where
  they're used - for example, ``Ridge(alpha=skrub.choose_float(0.01, 10.0, log=True))``.
  These choices can be applied to any part of your pipeline: model hyperparameters,
  encoder configurations, feature engineering techniques, or even to select between
  entirely different estimators or pipeline architectures. For detailed information,
  see :ref:`user_guide_data_ops_hyperparameter_tuning` and the example
  :ref:`sphx_glr_auto_examples_data_ops_13_choices.py`.

**Interactive Development**
  Build your pipeline step-by-step with immediate feedback on intermediate results.
  Each Data Op provides a preview of its output, allowing you to inspect the effect
  of each transformation as you add it to your pipeline. This feedback loop enables
  quick iteration and debugging without waiting for the full pipeline to run. Learn more
  about using previews in :ref:`user_guide_data_ops_using_previews`.

**Model and Pipeline Selection**
  Choose between different preprocessing strategies or machine learning algorithms
  within a single pipeline structure. With |choose_from|, you can select between
  completely different pipeline architectures - for example, choosing between a Ridge
  regression with optional scaling or a Random Forest with different parameter ranges.
  Choices can be nested and conditional, allowing you to build sophisticated
  pipelines that can automatically select the best combination of preprocessing
  techniques, encoders, and models through a single cross-validation process
  with |skb.make_randomized_search| or |skb.make_grid_search|. For advanced techniques,
  see :doc:`/modules/data_ops/validation/nesting_choices_choosing_pipelines`.

**Reusable Learners**
  Export any Data Ops pipeline as a "learner" with |skb.make_learner| - a specialized
  estimator that can be fitted, pickled, and deployed in production environments.
  Unlike traditional scikit-learn estimators that expect X and y matrices, learners
  take a dictionary of inputs (an "environment") where keys correspond to variable names
  in your DataOps plan. This powerful approach allows you to pass arbitrary inputs
  like file paths or URLs rather than requiring pre-loaded DataFrames. Learners
  encapsulate the entire pipeline, from raw data to predictions, making it easy
  to reproduce the same workflow on new data. This is particularly valuable for
  microservice deployments where you need to ensure that all preprocessing steps
  from development are precisely replicated in production. See :ref:` user_guide_data_ops_exporting`
  and the real-world deployment example :ref:`sphx_glr_auto_examples_data_ops_15_use_case.py`.

**Rich Reporting and Visualization**
  Generate detailed HTML reports that visualize pipeline structure, parameter
  choices, and performance metrics with |skb.full_report|. These reports provide insights into what's
  happening at each step of your pipeline and help identify bottlenecks or
  optimization opportunities. Learn more about documenting and debugging your
  Data Ops pipelines in :ref:`user_guide_documenting_data_ops_plan` and
  :ref:`user_guide_data_ops_evaluating_debugging_dataops`.

**scikit-learn Integration**
  Data Ops work seamlessly with any scikit-learn estimator or transformer,
  allowing you to leverage the extensive ecosystem of scikit-learn-compatible
  libraries within your Data Ops pipelines. While DataOps learners have a different
  interface than scikit-learn estimators (taking dictionaries rather than X and y matrices),
  they incorporate scikit-learn components internally and follow similar design principles.
  For information on applying scikit-learn estimators in Data Ops, see
  :ref:`user_guide_data_ops_applying_ml_estimators` and
  :ref:`user_guide_data_ops_applying_different_transformers`.

Getting Started
--------------

To start using Data Ops, begin by creating variables with |var| (or the
shortcuts |X| and |y|) and access the Data Ops
functionality through the ``.skb`` namespace. From there, you can build pipelines
using:

* Familiar pandas/polars operations on your DataOps variables
* |skb.apply| for applying scikit-learn or skrub transformers and models
* |skb.mark_as_X| and |skb.mark_as_y| for proper train/test splitting
* |skb.make_learner| to export your pipeline for reuse
* |skb.make_randomized_search| or |skb.make_grid_search| for tuning
* |skb.cross_validate| for evaluation
* |skb.full_report| for comprehensive pipeline analysis

For more details on these methods and the `.skb` namespace, see :ref:`user_guide_direct_access_ref`.

Data Ops empower you to build complex, maintainable machine learning pipelines
with less code, better organization, and more flexibility than traditional
approaches, while integrating seamlessly with the scikit-learn ecosystem.
They bridge the gap between exploratory data analysis and production-ready machine
learning pipelines, making the transition from development to deployment seamless.
The unique dictionary-based interface of learners offers greater flexibility than
standard scikit-learn pipelines, allowing your models to accept arbitrary inputs
beyond just DataFrames or arrays.

Common Usage Patterns
--------------------

**Basic Single-Table Pipeline**
  Start with a DataFrame, mark features and target, apply transformers, and train a model:
  ``data = skrub.var("data", df)``, ``X = data.drop("target").skb.mark_as_X()``,
  ``y = data["target"].skb.mark_as_y()``, ``X_vec = X.skb.apply(TableVectorizer())``,
  ``pred = X_vec.skb.apply(model, y=y)``. See a complete example in
  :ref:`sphx_glr_auto_examples_data_ops_11_data_ops_intro.py`.

**Multi-Table Processing**
  Process tables separately before joining: ``products_vec = products.skb.apply(vectorizer)``,
  ``agg_products = products_vec.groupby("basket_id").mean()``,
  ``features = baskets.merge(agg_products, on="basket_id")``. For a complete workflow,
  see :ref:`sphx_glr_auto_examples_data_ops_12_multiple_tables.py`.

**Hyperparameter Tuning**
  Add choices directly in the pipeline: ``Ridge(alpha=skrub.choose_float(0.01, 10.0))``,
  ``encoder=skrub.choose_from({"MinHash": MinHashEncoder(), "LSA": StringEncoder()})``.
  See detailed examples in :ref:`sphx_glr_auto_examples_data_ops_13_choices.py` and
  :ref:`user_guide_data_ops_hyperparameter_tuning`.

**Model Selection**
  Choose between different models: ``pred = skrub.choose_from({"ridge": ridge_pipeline, "rf": rf_pipeline}).as_data_op()``.
  Learn more about this technique in :ref:`user_guide_data_ops_nesting_choices`.

**Production Deployment**
  Export and reuse: ``learner = pipeline.skb.make_learner(fitted=True)``,
  ``with open("model.pkl", "wb") as f: pickle.dump(learner, f)``,
  ``predictions = loaded_learner.predict({"data": new_data})``. For a real-world
  deployment example, see :ref:`sphx_glr_auto_examples_data_ops_15_use_case.py`.

**Custom Transformations with Deferred Functions**
  Embed arbitrary Python code in your DataOps pipeline using |deferred| functions:
  ``@skrub.deferred def custom_transform(df): return df.assign(new_col=df['col'].apply(complex_logic))``.
  Deferred functions enable you to use Python control flow (if/for/while) and other operations that
  would normally execute immediately, by delaying their execution until the pipeline runs. Inside
  a deferred function, DataOps are evaluated to their actual values, allowing you to treat them as
  regular Python objects (e.g., pandas DataFrames). This is particularly useful for complex
  transformations that aren't easily expressed with standard methods, like text processing, feature
  engineering with custom logic, or operations that need to iterate over columns or rows. Learn more
  in :doc:`/modules/data_ops/basics/control_flow`.


Learn More
---------

For a comprehensive exploration of Data Ops capabilities, refer to the following resources:

* Complete Data Ops documentation: :ref:`user_guide_data_ops_index`
* Basic concepts: :ref:`user_guide_data_ops_intro`
* Building ML pipelines: :ref:`user_guide_data_ops_applying_ml_estimators`
* Tuning and validation: :ref:`user_guide_data_ops_tuning_validating_dataops`
* Gallery of examples: :ref:`data_ops_examples_ref`
