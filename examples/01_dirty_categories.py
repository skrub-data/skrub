"""
.. _example_introducing_encoding:

=======================================================
Introducing skrub's encoding methods on a dirty dataset
=======================================================

In this example, we want to predict wages using the
`employee salaries <https://www.openml.org/d/42125>`_ dataset.

Let's dive right in!


.. |TableVectorizer| replace::
    :class:`~skrub.TableVectorizer`

.. |Pipeline| replace::
    :class:`~sklearn.pipeline.Pipeline`

.. |make_pipeline| replace::
    :func:`~sklearn.pipeline.make_pipeline`

.. |cross_validate| replace::
    :func:`~sklearn.model_selection.cross_validate`

.. |OneHotEncoder| replace::
    :class:`~sklearn.preprocessing.OneHotEncoder`

.. |ColumnTransformer| replace::
    :class:`~sklearn.compose.ColumnTransformer`

.. |GapEncoder| replace::
    :class:`~skrub.GapEncoder`

.. |MinHashEncoder| replace::
    :class:`~skrub.MinHashEncoder`

.. |HGBR| replace::
    :class:`~sklearn.ensemble.HistGradientBoostingRegressor`

.. |SimilarityEncoder| replace::
    :class:`~skrub.SimilarityEncoder`
"""

###############################################################################
# Prediction
# ----------
#
#  .. note::
#     TLDR: use
#
#     .. code-block:: python
#        pipeline = make_pipeline(TableVectorizer(), HistGradientBoostingRegressor())
#        pipline.fit(X, y)
#
#     as a generic and robust pipeline for your tabular learning tasks.
#
# Let's first retrieve the dataset:

from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()

dataset.description

###############################################################################
# Alias *X*, the descriptions of employees (our input data), and *y*,
# the annual salary (our target column):

X = dataset.X
y = dataset.y

###############################################################################
# And carry out some basic preprocessing

# Overload `employee_position_title` with 'underfilled_job_title',
# as the latter gives more accurate job titles when specified

X["employee_position_title"] = X["underfilled_job_title"].fillna(
    X["employee_position_title"]
)
X.drop(labels=["underfilled_job_title"], axis="columns", inplace=True)

X

###############################################################################
# Let's extract a sample to have a more focused view: we'll limit our
# search to employees with position title containing "Fire".

sample = X[X["employee_position_title"].str.contains("Fire")].sample(
    n=10, random_state=5
)

sample

###############################################################################
# We observe a few things:
# - We have diverse columns: binary ('gender'), numerical
#   ('employee_annual_salary'), categorical ('department', 'department_name',
#   'assignment_category'), datetime ('date_first_hired') and dirty categories
#   ('employee_position_title', 'division').
#
# Now, without much more investigation, we can already build a machine-learning
# pipeline and train it:

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer

pipeline = make_pipeline(TableVectorizer(), HistGradientBoostingRegressor())
pipeline.fit(X, y)

###############################################################################
# What just happened there?
# -------------------------
#
# First, it did not raise any errors, yay!
# Let's explore the internals of our encoder, the |TableVectorizer|:

from pprint import pprint

# Recover the TableVectorizer from the pipeline
tv = pipeline.named_steps["tablevectorizer"]

pprint(tv.transformers_)

###############################################################################
# We observe it has automatically assigned an appropriate encoder to
# corresponding columns.
# For example, it classified the columns 'gender', 'department',
# 'department_name' and 'assignment_category' as low cardinality
# string variables.
# Two remarkable things:, it has affected a |GapEncoder| to the columns
# `employee_position_title` and `division`, and a |DatetimeEncoder| to the
# 'date_first_hired' column.
#
# The |GapEncoder| is a powerful encoder that can handle dirty
# categorical columns.
# The |DatetimeEncoder| can encode datetime columns for machine learning.
#
# We will explore those in the next sections.
#

# TODO

###############################################################################
# Conclusion
# ----------
#
# In this example, we motivated the need for a simple machine learning
# pipeline, which we built using the |TableVectorizer| and a
# |HistGradientBoostingRegressor|.
#
# We explored the main encoding features implemented in the *skrub* library.
