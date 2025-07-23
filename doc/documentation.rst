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

For those tasks, skrub does not replace pandas or polars. Instead, it
leverages the dataframe libraries to provide more high-level building blocks that
perform the data preprocessing steps that are typically needed in a machine learning
pipeline.

This guide demonstrates how to resolve various issues using Skrub's features.
See the examples section for full code snippets.


.. include:: includes/big_toc_css.rst

.. toctree::
   :maxdepth: 2

   userguide_tablereport
   userguide_encoders
   userguide_datetimes
   userguide_tablevectorizer
   userguide_data_cleaning
   userguide_selectors
   userguide_data_ops
   userguide_data_ops_ml_pipeline
   userguide_data_ops_validation
   userguide_joining_tables
   userguide_utils
