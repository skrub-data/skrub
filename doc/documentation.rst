.. _user_guide:

User Guide
=============================================================
``skrub`` is a library that eases the preparation of tabular data in dataframes
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

The transformations implemented by ``skrub`` are *stateful*: ``skrub``
records the transformations that were applied to the training data and replays them
when the pipeline makes predictions on unseen data: this is essential to prevent
data leakage and ensure generalization.

This guide showcases how to solve various problems using the features of ``skrub``.
Short code snippets are provided to explain each operation. Additional examples
are shown in the docstrings and in the ``examples`` section of the documentation.


.. include:: includes/big_toc_css.rst

.. toctree::
   :maxdepth: 2

   userguide_tablereport
   userguide_data_cleaning
   userguide_tablevectorizer
   userguide_encoders
   userguide_datetimes
   userguide_selectors
   userguide_joining_tables
   userguide_utils
