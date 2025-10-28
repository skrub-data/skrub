.. |TableReport| replace:: :class:`~skrub.TableReport`
.. |Cleaner| replace:: :class:`~skrub.Cleaner`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |tabular_pipeline| replace:: :func:`~skrub.tabular_pipeline`

.. _user_guide:

User Guide
==========
Skrub is a library that eases machine learning with dataframes, from exploring
dataframes to validating a machine-learning pipeline.

The |TableReport| is a powerful data exploration tool, which can be followed by
data sanitization and feature engineering tools in the |Cleaner| and |TableVectorizer|.
The |tabular_pipeline| combines the two to build a strong baseline for dataframes.

The skrub :ref:`column-level encoders<user_guide_encoders_index>` can be tweaked by the user for more
specific needs.
Various :ref:`multi-column transformers <user_guide_building_pipeline_index>` and the :ref:`selectors API<user_guide_selectors>`
provide a high degree of control over which columns should be modified.

More complex, multi-table scenarios can make use of the skrub :ref:`Data Ops <user_guide_data_ops_index>`,
which enable constructing and validating pipelines that involve
multiple dataframes and hyperparameter tuning.

Skrub does not replace pandas or polars. Instead, it
leverages the dataframe libraries to provide more high-level building blocks that
perform the data preprocessing steps that are typically needed in a machine learning
pipeline.



.. include:: includes/big_toc_css.rst

.. toctree::
   :maxdepth: 3

   exploring_a_dataframe
   default_wrangling
   column_level_featurizing
   multi_column_operations
   data_ops
   configuration_and_utils
   joining_dataframes
