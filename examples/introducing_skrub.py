"""
.. example_intro_skrub:

======================================================================
skrub: Tackling Non-Normalized Categories In Machine Learning Settings
======================================================================

Handling strings that represent categories often requires substantial data
preparation, especially when they contain various morphological variants
emanating from manual input or diverse data sources.

..note:
    A variable is categorical when it can take on one of a
    fixed and limited (discrete) number of possible values,
    as opposed to continuous variables and free text,
    which can take a very large or infinite number of unique values.

Typically, we encounter several types of morphological variants in categories:
- **Typos**: "France" vs "Frqnce"
- **Abbreviations**: "USA" vs "United States of America"
- **Duplicated values**: "France/FR"
- **Alternate values**: "Bordeaux, FR" & "Bordeaux"
- **Variations**: "waiter" & "waitress"

While some of these variations carry meaning, others do not. For instance,
"Frqnce" is simply misspelled, while "waiter" and "waitress" represent the same
profession, performed by individuals of different genders. In a salary
prediction context, the latter variation is likely significant for model
accuracy, while the former is irrelevant.

Using the `employee salaries <https://www.openml.org/d/42125>`_ dataset,
we aim to predict wages.

This guide offers an overview of *skrub*, a library designed to handle
dirty categorical data in machine learning settings.

We will explore the following topics:

1. Comparing the performance of a traditional encoder, the |OneHotEncoder|,
   with the encoders provided by the *skrub* library
2. Introducing a simpler approach to assembling a machine-learning pipeline
3. Demonstrating how to join and augment dirty tables effectively
4. Exploring methods to clean data when identified similarities are inconsequential


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

.. |make_column_transformer| replace::
    :func:`~sklearn.compose.make_column_transformer`

.. |GapEncoder| replace::
    :class:`~skrub.GapEncoder`

.. |MinHashEncoder| replace::
    :class:`~skrub.MinHashEncoder`

.. |HGBR| replace::
    :class:`~sklearn.ensemble.HistGradientBoostingRegressor`

.. |SimilarityEncoder| replace::
    :class:`~skrub.SimilarityEncoder`

.. |fuzzy_join| replace::
    :func:`~skrub.fuzzy_join`

.. |FeatureAugmenter| replace::
    :class:`~skrub.FeatureAugmenter`

.. |deduplicate| replace::
    :func:`~skrub.deduplicate`
"""

###############################################################################
# Prediction in the presence of dirty categories
# ----------------------------------------------
#
# Let's first retrieve the dataset:
from skrub.datasets import fetch_employee_salaries

employee_salaries = fetch_employee_salaries()

employee_salaries.description

###############################################################################
# Alias *X*, the descriptions of employees (our input data), and *y*,
# the annual salary (our target column):
X = employee_salaries.X
y = employee_salaries.y

###############################################################################
# And carry out some basic preprocessing

# Overload `employee_position_title` with `underfilled_job_title`,
# as the latter gives more accurate job titles when specified
X["employee_position_title"] = X["underfilled_job_title"].fillna(
    X["employee_position_title"]
)
X.drop(labels=["underfilled_job_title"], axis="columns", inplace=True)

X

###############################################################################
# In this dataset, we observe a few interesting columns, but one stands out
# especially: the `employee_position_title` column.
# This is probably one of the more important columns in regard to the salary.
# However, its content is not normalized: we have different numbers
# (e.g. "I", "II", "III", etc.), which could be considered ranks.
#
# Let's investigate that further by picking a sample:

sample = X[X["employee_position_title"].str.contains("Fire")].sample(
    n=10, random_state=5
)

# We'll keep only the column we're interested in
sample = sample[["employee_position_title"]]

sample

# We observe some additional information (such as "(Recruit)"), as well as
# some variations of a same (or at least very similar) job title
# ("Fire/Rescue" vs "Firefighter/Rescuer").
#
# Taking into account the definition of what's dirty data we had at the
# beginning of this example, we can consider that **the column
# "employee_position_title" is dirty!**

###############################################################################
# Integrating dirty variables in the machine-learning pipeline
# ------------------------------------------------------------
#
# An encoder is required in order to turn categorical columns such as
# "department_name" or "employee_position_title" into numerical
# representations suited for machine learning.
#
# Classical encoders include, among others, the |OneHotEncoder| and
# the |LabelEncoder|, though one-hot is the most popular.
#
# In :ref:`_exemple_explaining_similarity_encoding`, we explore
# the mechanisms and interpretation of one-hot, as well as the encoders
# provided by the *skrub* library.
#
# Here, we will focus solely on the performance comparison.

###############################################################################
# Performance comparison
# ......................
#
# In order to compare the performance of the aforementioned encoders,
# we will build a machine learning pipeline.
#
# We specify which columns are to be encoded with which transformer
# with to a |ColumnTransformer| (for simplicity,
# here we use the |make_column_transformer| function).
# We then loop over the different encoding methods,
# and instantiate a new |Pipeline| with |make_pipeline| each time,
# and |cross_validate| it to get the most accurate score:
#
# We will use a |HGBR| as our learner,
# which is a good predictor for heterogeneous data
#
# .. note:
#    You might need to require the experimental feature for scikit-learn
#    versions earlier than 1.0 with:
#    ```py
#    from sklearn.experimental import enable_hist_gradient_boosting
#    ```
#

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_validate
from skrub import (
    SimilarityEncoder,
    TargetEncoder,
    MinHashEncoder,
    GapEncoder,
)

one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

durations = dict()
scores = dict()

for method in {
    one_hot,
    SimilarityEncoder(),
    TargetEncoder(handle_unknown="ignore"),
    GapEncoder(n_components=50),
    MinHashEncoder(n_components=100),
}:
    name = method.__class__.__name__  # Extract the encoder name
    encoder = make_column_transformer(
        (one_hot, ["gender", "department_name", "assignment_category"]),
        ("passthrough", ["year_first_hired"]),
        (method, ["employee_position_title"]),
        remainder="drop",
    )
    pipeline = make_pipeline(encoder, HistGradientBoostingRegressor())

    results = cross_validate(pipeline, X, y)
    scores[name] = results["test_score"]
    durations[name] = results["fit_time"]

###############################################################################
# Next up: plot the results

import pandas as pd
import matplotlib.pyplot as plt
from seaborn import boxplot

_, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 6))

boxplot(data=pd.DataFrame(scores), orient="h", ax=ax1)
ax1.set_xlabel("Prediction accuracy", size=20)
[t.set(size=20) for t in ax1.get_yticklabels()]

boxplot(data=pd.DataFrame(durations), orient="h", ax=ax2)
ax2.set_xlabel("Computation time (s)", size=20)
[t.set(size=20) for t in ax2.get_yticklabels()]

plt.tight_layout()

###############################################################################
# The clear trend is that encoders grasping similarities between categories
# (|SimilarityEncoder|, |MinHashEncoder|, and |GapEncoder|)
# perform better than those discarding it.
#
# |SimilarityEncoder| is the best performer, but it is less scalable on big
# data than the |MinHashEncoder| and |GapEncoder|. The most scalable encoder is
# the |MinHashEncoder|. On the other hand, the |GapEncoder| has the benefit of
# providing interpretable features
# (see :ref:`_example_interpreting_gap_encoder` for more details).

###############################################################################
# A simpler way: automatic vectorization
# --------------------------------------
#
# The code to assemble a column transformer is a bit tedious. We will
# now explore a simpler, automated way of encoding the data.
#
# Let's start again from the raw data:

employee_salaries = fetch_employee_salaries()
X = employee_salaries.X
y = employee_salaries.y

# Overload `employee_position_title` with `underfilled_job_title`,
# as the latter gives a more accurate job title when specified
X["employee_position_title"] = X["underfilled_job_title"].fillna(
    X["employee_position_title"]
)
X.drop(labels=["underfilled_job_title"], axis="columns", inplace=True)

###############################################################################
# We still have a complex and heterogeneous dataframe:
X

###############################################################################
# Using the TableVectorizer in a supervised-learning pipeline
# -----------------------------------------------------------
#
# Like we did with the |ColumnTransformer|, we can assemble the
# |TableVectorizer| in a |Pipeline| with a |HGBR|

from skrub import TableVectorizer

pipeline = make_pipeline(TableVectorizer(), HistGradientBoostingRegressor())

###############################################################################
# Let's perform a cross-validation to see how well this model predicts:

import numpy as np
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, X, y, scoring="r2")

print(f"scores={scores}")
print(f"mean={np.mean(scores)}")
print(f"std={np.std(scores)}")

###############################################################################
# The prediction performed here is as good as the earlier method,
# with the added benefit that this code is much simpler as it does not involve
# specifying columns manually.

###############################################################################
# Inspecting the features created
# ...............................
#
# Let's fit the |TableVectorizer| again, so we can analyze its mechanisms.

from sklearn.model_selection import train_test_split

table_vec = TableVectorizer()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train_enc = table_vec.fit_transform(X_train, y_train)
X_test_enc = table_vec.transform(X_test)

###############################################################################
# The encoded data, ``X_train_enc`` and ``X_test_enc`` are numerical arrays:

X_train_enc

###############################################################################
# They have more columns than the original dataframe, but reasonably so:

print(f"{X_train.shape=}")
print(f"{X_train_enc.shape=}")

###############################################################################
# The |TableVectorizer| assigns a transformer for each column.
# We can inspect this choice:

from pprint import pprint

pprint(table_vec.transformers_)

###############################################################################
# This is what is being passed to the |ColumnTransformer| under the hood.
# If you're familiar with how it works, it should be very intuitive.

# We can notice it classified the columns 'gender' and 'assignment_category'
# as low cardinality string variables.
# A |OneHotEncoder| will be applied to these columns.
#
# Next, we can have a look at the encoded feature names.
#
# Before encoding:

X.columns.to_list()

###############################################################################
# After encoding (we only plot the first 8 feature names):

feature_names = table_vec.get_feature_names_out()
feature_names[:8]

###############################################################################
# As we can see, it gave us interpretable columns.
# This is because we used the |GapEncoder| on the column 'division',
# which was classified as a high cardinality string variable.
#
# In total, we have a reasonable number of encoded columns:

len(feature_names)

###############################################################################
# Joining and augmenting dirty tables
# -----------------------------------
#
# Example :ref:`sphx_glr_auto_examples_04_fuzzy_joining.py` shows how to
# join multiple dirty tables with fuzzy matching.
# It introduces the |fuzzy_join| function, as well as the |FeatureAugmenter|.
#
#
# Example :ref:`sphx_glr_auto_examples_06_ken_embeddings.py` shows how to
# augment and enrich your data with embeddings extracted from Wikipedia.

###############################################################################
# Cleaning dirty tables
# ---------------------
#
# Example :ref:`sphx_glr_auto_examples_05_deduplication.py` shows how to
# clean dirty variables with |deduplicate|.

###############################################################################
# Conclusion
# ----------
#
# In this example, we motivated the need for methods for handling dirty data
# in machine learning settings.
#
# We explored the main features of the *skrub* library.
#
# Reading the other examples is greatly encouraged!
