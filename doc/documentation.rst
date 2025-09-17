.. _user_guide:

User Guide
==========
Skrub is a library that eases machine learning with dataframes
for machine learning.

Starting from rich, complex data stored in one or several dataframes, it helps
performing the data wrangling necessary to produce a numeric array that is fed
to a machine-learning model. This wrangling comprises joining tables (possibly
with inexact matches), parsing structured data such as datetimes from text,
and extracting numeric features from non-numeric data.

Skrub does not replace pandas or polars. Instead, it
leverages the dataframe libraries to provide more high-level building blocks that
perform the data preprocessing steps that are typically needed in a machine learning
pipeline.

This guide demonstrates how to resolve various issues using Skrub's features.
See the examples section for full code snippets.


.. include:: includes/big_toc_css.rst

.. toctree::
   :maxdepth: 3

   exploring_a_dataframe
   default_wrangling
   column_level_featurizing
   multi_column_operations
   data_ops
   configuration_and_datasets
   joining_dataframes
