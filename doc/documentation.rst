User guide
===========

Skrub facilitates preparing tables for machine learning.

Starting from rich, complex data stored in one or several dataframes, it helps
perform the data wrangling necessary to produce a numeric array that can be fed
to a machine-learning model. This wrangling comprises joining tables (possibly
with inexact matches), parsing text into structured data such as dates,
extracting numeric features, etc.

For those tasks, Skrub does not replace a dataframe library. Instead, it
leverages Polars or Pandas to provide more high-level building blocks that are
typically needed in a machine-learning pipeline.

Crucially, the transformations implemented by Skrub are *stateful*: Skrub
records the transformation that was applied to the training data and replays the
same operations when the pipeline is applied to make predictions on unseen
data. Implementing data-wrangling steps as transformers that can be fitted is
essential to prevent data leakage and ensure generalization.

.. topic:: Skrub highlights:

 - facilitates separating the train and test operations, allowing to tune
   preprocessing steps to the data and improving the generalization of tabular
   machine-learning models.

 - enables statistical and imperfect assembly, as machine-learning models
   can typically retrieve signals even in noisy data.

|

.. include:: includes/big_toc_css.rst

.. toctree::
   :maxdepth: 2

   end_to_end_pipeline
   encoding
   assembling
   cleaning
