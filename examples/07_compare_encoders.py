"""
.. _example_compare_encoders

========================================================
Understanding and comparing categorical encoding methods
========================================================

In this example, we compare various categorical encoders:
a classical encoding method, the |OneHotEncoder|,
and those implemented in *skrub*, that is, the |SimilarityEncoder|,
the |GapEncoder| and the |MinHashEncoder|.

.. note:
    Reading :ref:`sphx_glr_auto_examples_01_dirty_categories.py` up to
    *Performance comparison* is highly recommended before reading this example.
    It's also recommended to continue reading after finishing this example!

We use the `employee salaries <https://www.openml.org/d/42125>`_ dataset
for illustration purposes.


.. |OneHotEncoder| replace::
    :class:`~sklearn.preprocessing.OneHotEncoder`

.. |SimilarityEncoder| replace::
    :class:`~skrub.SimilarityEncoder`

.. |GapEncoder| replace::
    :class:`~skrub.GapEncoder`

.. |MinHashEncoder| replace::
    :class:`~skrub.MinHashEncoder`
"""

###############################################################################
# Encoding dirty categorical variables
# ------------------------------------
#
# Let's first retrieve the dataset:

from skrub.datasets import fetch_employee_salaries

employee_salaries = fetch_employee_salaries()

employee_salaries.description

###############################################################################
# Alias X, the descriptions of employees (our input data), and y,
# the annual salary (our target column):

X = employee_salaries.X
y = employee_salaries.y

###############################################################################
# Let's carry out some basic preprocessing:

# Overload `employee_position_title` with `underfilled_job_title`,
# as the latter gives more accurate job titles when specified
X["employee_position_title"] = X["underfilled_job_title"].fillna(
    X["employee_position_title"]
)
X.drop(labels=["underfilled_job_title"], axis="columns", inplace=True)

X

###############################################################################
# As we saw in :ref:`sphx_glr_auto_examples_01_dirty_categories.py`, the
# column `employee_position_title` is dirty.
#
# A classical encoding method is one-hot encoding (in the form of the
# |OneHotEncoder|): it works by creating a column for each category, and
# filling it with 1 if the category is present in the sample, and 0 otherwise.
#
# Let's visualize that!
#
# We'll handpick a set of 10 samples which have a similar job name,
# that is, the position title contains the word "Fire"

sample = X[X["employee_position_title"].str.contains("Fire")].sample(
    n=10, random_state=5
)

# We'll keep only the column we're interested in
sample = sample[["employee_position_title"]]

sample

###############################################################################
# Next up: import the |OneHotEncoder| and apply it to our sample:

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

one_hot = OneHotEncoder(sparse_output=False)
sample_enc_ohe = one_hot.fit_transform(sample)

# (make it look nice in Jupyter by wrapping the resulting array in a DataFrame)
pd.DataFrame(
    sample_enc_ohe,
    columns=one_hot.categories_[0],
    index=sample["employee_position_title"],
)

###############################################################################
# The indices are the samples (the lines in the dataset) and the columns are
# the unique categories.
# As outlined earlier, ones are located where both match.
#
# One-hot is a powerful encoding when the data is clean and independent,
# but in the case of dirty data, it is not for a few reasons
# resulting from the presence of rare categories:
# - Dimensionality explodes
# - It's likely that some will not be seen during fitting
# - All categories are equidistant
#
# *skrub* tries to address these issues.
# For the problems raised by the use of the one-hot encoder,
# it provides a few encoders, which we'll explore one by one.
# We will also compare their performance in a following section.

###############################################################################
# An improvement to the OneHotEncoder: the SimilarityEncoder
# ..........................................................
#
# The |SimilarityEncoder| is a generalization of the |OneHotEncoder|, which
# uses a string similarity metric to encode the similarity between categories.
# As we did for the one-hot, let's visualize what that means!

from skrub import SimilarityEncoder

sim_enc = SimilarityEncoder()
sample_enc_sim = sim_enc.fit_transform(sample)

# (make it look nice in Jupyter by wrapping the resulting array in a DataFrame)
pd.DataFrame(
    sample_enc_sim,
    columns=sim_enc.categories_[0],
    index=sample["employee_position_title"],
)

###############################################################################
# We can see that instead of having zeros where the categories don't exactly
# match, we have a similarity score between 0 and 1.
# The representation is thus more meaningful. We see that
# "Firefighter/Rescuer II" has a ~93% similarity with
# "Firefighter/Rescuer III", while it has a 20% similarity with
# "Fire/Rescue Battalion Chief".
#
# We can better see that by throwing it in a heatmap:

import matplotlib.pyplot as plt
from seaborn import heatmap


ax = plt.figure(figsize=(6, 6))
heatmap(
    sample_enc_sim,
    xticklabels=sim_enc.categories_[0],
    yticklabels=sample["employee_position_title"],
    ax=ax,
)
plt.tight_layout()

###############################################################################
# While this is a cool result, it suffers from the high computational cost
# of calculating the similarities.
# It really only addresses the independence of encoded values compared to
# one-hot.
#
# Let's plot the results, so we can better comprehend what similarity means
# in this context.
#
# To do that, we'll first cast the similarity matrix to pairs of points
# using a |MDS| (Multi-Dimensional Scaling) algorithm.

from sklearn.manifold import MDS

# The MDS works on square matrices, so we'll get the unique values in our
# sample, and then compute the similarity matrix
unique_values = sample.drop_duplicates()
unique_values_enc = SimilarityEncoder().fit_transform(unique_values)
# Ensure the output is a square matrix
assert unique_values_enc.shape[0] == unique_values_enc.shape[1]
# Invert the similarity matrix (thus we get a dissimilarity matrix)
dissimilarities = 1 - unique_values_enc

mds = MDS(dissimilarity="precomputed", n_init=10, random_state=0)
points = mds.fit_transform(dissimilarities)

print(f"{unique_values_enc.shape=}")
print(f"{points.shape=}")

###############################################################################
# We can now plot the points:

f, ax = plt.figure()
ax.scatter(x=points[:, 0], y=points[:, 1])
# Add the labels
for i, (x, y) in enumerate(points):
    ax.text(
        x=x,
        y=y,
        s=unique_values["employee_position_title"][x],
        fontsize=8,
    )
ax.set_title("MDS representation of the similarity matrix")

###############################################################################
# Thanks to this representation, we can clearly see the distance between
# categories.

###############################################################################
# The mechanism and output of the |SimilarityEncoder| are easy to understand,
# which is why we introduce it first, but in practice,
# it scales terribly to large datasets,
# making the |GapEncoder| a more appropriate solution.

###############################################################################
# Fixing the remaining problems: the |GapEncoder|
# .............................................
#
# The |GapEncoder| (Gamma-Poisson Encoder) has a different approach to the
# problem.
# It constructs an encoding based on a set of latent categories,
# which are built by capturing combinations of substrings that frequently
# co-occur in the data.

from skrub import GapEncoder

# We want to model 10 topics
gap_enc = GapEncoder(n_components=10, random_state=0)

# Fit it on all our data, not just our sample
sample_enc_gap = gap_enc.fit_transform(X)

sample_enc_gap.shape

###############################################################################
# Interpreting its output
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# The encoded vectors correspond to the activation of each latent topic.
#
# To interpret these latent topics, we can select, for each one, a few labels
# from the input data with the highest activations.

# We select 3 labels to summarize each topic
topic_labels = gap_enc.get_feature_names_out(n_labels=3)
for k, labels in enumerate(topic_labels):
    print(f"Topic n°{k}: {labels}")

###############################################################################
# As expected, topics capture labels that frequently co-occur. For instance,
# the labels "firefighter", "rescuer", "rescue" appear together in
# "Firefighter/Rescuer III", or "Fire/Rescue Lieutenant".
#
# This enables us to understand the encoding of different samples

n_samples = 10

encoded_labels = gap_enc.transform(X[["employee_position_title"]][:n_samples])
plt.figure(figsize=(8, 6))
plt.imshow(encoded_labels)
plt.xlabel("Latent topics", size=12)
plt.xticks(range(gap_enc.n_components), labels=topic_labels, rotation=40, ha="right")
plt.ylabel("Samples", size=12)
plt.yticks(
    range(n_samples),
    labels=X[["employee_position_title"]][:n_samples].to_numpy().flatten(),
)
plt.colorbar().set_label(label="Topic activations", size=12)
plt.tight_layout()

###############################################################################
# As we can see, each dirty category encodes on a small number of topics,
# These can thus be reliably used to summarize each topic, which are in
# effect latent categories captured from the data.


###############################################################################
# Scaling to the extreme: the MinHashEncoder
# ..........................................
#
# The |MinHashEncoder| uses the
# `multi-function MinHash <https://en.wikipedia.org/wiki/MinHash>`_
# algorithm to encode the similarities in a multidimensional space.
# These embeddings have nice properties which are relevant for our problem.

# TODO
