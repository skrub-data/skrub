"""
.. _example_interpreting_gap_encoder:

==========================================
Feature interpretation with the GapEncoder
==========================================

In this notebook, we will explore the output and inner workings of the
|GapEncoder|, one of the `high cardinality categorical encoders <https://inria.hal.science/hal-02171256v4>`_
provided by skrub.

.. |GapEncoder| replace::
     :class:`~skrub.GapEncoder`

.. |SimilarityEncoder| replace::
     :class:`~skrub.SimilarityEncoder`
"""

###############################################################################
# The |GapEncoder| is scalable and interpretable in terms of
# finding latent categories, as we will show.
#
# First, let's retrieve the
# `employee salaries dataset <https://www.openml.org/d/42125>`_:

from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()

# Alias X and y
X, y = dataset.X, dataset.y

X

###############################################################################
# Encoding job titles
# -------------------
#
# Let's look at the job titles, the column containing dirty data we want to encode:
#
# .. topic:: Note:
#
#   Dirty data, as opposed to clean, are all non-curated categorical
#   columns with variations such as typos, abbreviations, duplications,
#   alternate naming conventions etc.
#

X_dirty = X["employee_position_title"]

###############################################################################
# Let's have a look at a sample of the job titles:

X_dirty.sort_values().tail(15)

###############################################################################
# Then, we create an instance of the |GapEncoder| with 10 components.
# This means that the encoder will attempt to extract 10 latent topics
# from the input data:

from skrub import GapEncoder

enc = GapEncoder(n_components=10, random_state=1)

###############################################################################
# Finally, we fit the model on the dirty categorical data and transform it
# in order to obtain encoded vectors of size 10:

X_enc = enc.fit_transform(X_dirty)
X_enc.shape

###############################################################################
# Interpreting encoded vectors
# ----------------------------
#
# The |GapEncoder| can be understood as a continuous encoding
# on a set of latent topics estimated from the data. The latent topics
# are built by capturing combinations of substrings that frequently
# co-occur, and encoded vectors correspond to their activations.
# To interpret these latent topics, we select for each of them a few labels
# from the input data with the highest activations.
# In the example below we select 3 labels to summarize each topic.

topic_labels = enc.get_feature_names_out(n_labels=3)
for k, labels in enumerate(topic_labels):
    print(f"Topic n°{k}: {labels}")

###############################################################################
# As expected, topics capture labels that frequently co-occur. For instance,
# the labels "firefighter", "rescuer", "rescue" appear together in
# "Firefighter/Rescuer III", or "Fire/Rescue Lieutenant".
#
# We can now understand the encoding of different samples.

import matplotlib.pyplot as plt

encoded_labels = enc.transform(X_dirty[:20])
plt.figure(figsize=(8, 10))
plt.imshow(encoded_labels)
plt.xlabel("Latent topics", size=12)
plt.xticks(range(0, 10), labels=topic_labels, rotation=50, ha="right")
plt.ylabel("Data entries", size=12)
plt.yticks(range(0, 20), labels=X_dirty[:20].to_numpy().flatten())
plt.colorbar().set_label(label="Topic activations", size=12)
plt.tight_layout()
plt.show()

###############################################################################
# As we can see, each dirty category encodes on a small number of topics,
# These can thus be reliably used to summarize each topic, which are in
# effect latent categories captured from the data.
#
# Conclusion
# ----------
#
# In this notebook, we have seen how to interpret the output of the
# |GapEncoder|, and how it can be used to summarize categorical variables
# as a set of latent topics.
