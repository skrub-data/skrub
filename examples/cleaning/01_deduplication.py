"""
.. _examples_deduplication:

===============================================
Cleaning misspelled categories with deduplicate
===============================================

Real world datasets often come with slight misspellings in the category
names, for instance if the category is manually input. Such misspellings
break many data-analysis steps that require exact matching, such as a
'GROUP BY'.

Merging the multiple variants of the same category or entity is known as
*deduplication*. It is implemented in skrub by the |deduplicate| function.

Deduplication relies on *unsupervised learning*, to find structures in
data without providing explicit labels/categories of the data a-priori.
Specifically, clustering of the distance between strings can be used to
find clusters of strings that are similar to each other (e.g. differ only
by a misspelling) and hence, gives us an easy tool to flag potentially
misspelled category names in an unsupervised manner.


.. |deduplicate| replace::
    :func:`~skrub.deduplicate`
"""

###############################################################################
# An example dataset
# -------------------
#
# Let's take the following example:
# as a data scientist, our job is to analyze the data from a hospital ward.
# We notice that most of the cases involve the prescription
# of one of three different medications:
# "Contrivan", "Genericon", or "Zipholan".
# However, data entry is manual and - either because the prescribing doctor's
# handwriting was hard to decipher, or due to mistakes during data input -
# there are multiple spelling mistakes in the dataset.
#
# Let's generate some example data that demonstrate this.

import numpy as np

from skrub.datasets import make_deduplication_data

duplicated_names = make_deduplication_data(
    examples=["Contrivan", "Genericon", "Zipholan"],  # our three medication names
    entries_per_example=[500, 100, 1500],  # their respective number of occurrences
    prob_mistake_per_letter=0.05,  # 5% probability of typo per letter
    random_state=42,  # set seed for reproducibility
)
# we extract the unique medication names in the data & how often they appear
unique_examples, counts = np.unique(duplicated_names, return_counts=True)

# and build a series out of them
import pandas as pd

ex_series = pd.Series(counts, index=unique_examples)

ex_series.head()

###############################################################################
# Visualize the data
# ------------------

import matplotlib.pyplot as plt

ex_series.plot.barh(figsize=(10, 15))
plt.xlabel("Medication name")
plt.ylabel("Counts")

###############################################################################
# We can now see clearly the structure of the data:
# the three original medications are the most common ones,
# but there are many spelling mistakes and hence
# many slight variations of the original names.
#
# The idea is to use the fact that the string-distance of each misspelled
# medication name will be closest to either the correctly or incorrectly
# spelled original medication name - and therefore form clusters.

###############################################################################
# Visualizing the pair-wise distance between all names
# ----------------------------------------------------
#
# Below, we use a heatmap to visualize the pairwise-distance between medication
# names. A darker color means that two medication names are closer together
# (i.e. more similar), a lighter color means a larger distance.
#
# We can see that we are dealing with three clusters - the original medication
# names and their misspellings that cluster around them.

from scipy.spatial.distance import squareform

from skrub import compute_ngram_distance

ngram_distances = compute_ngram_distance(unique_examples)
square_distances = squareform(ngram_distances)

import seaborn as sns

fig, axes = plt.subplots(1, 1, figsize=(12, 12))
sns.heatmap(
    square_distances, yticklabels=ex_series.index, xticklabels=ex_series.index, ax=axes
)

###############################################################################
# Deduplication: suggest corrections of misspelled names
# ------------------------------------------------------
#
# The |deduplicate| function uses clustering based on
# string similarities to group duplicated names
#
# The number of clusters will need some adjustment depending on the data.
# If no fixed number of clusters is given, |deduplicate| tries to set it
# automatically via the
# `silhouette score <https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient>`_.

from skrub import deduplicate

deduplicated_data = deduplicate(duplicated_names)

###############################################################################
# We can visualize the distribution of categories in the deduplicated data:

deduplicated_unique_examples, deduplicated_counts = np.unique(
    deduplicated_data, return_counts=True
)
deduplicated_series = pd.Series(deduplicated_counts, index=deduplicated_unique_examples)

deduplicated_series.plot.barh(figsize=(10, 15))
plt.xlabel("Medication name")
plt.ylabel("Counts")

###############################################################################
# Here, we can rely on the silhouette score to find the ideal number of
# clusters and correct the spelling mistakes.
#
# In practice however, the translation/deduplication will often be imperfect
# and thus require some tweaks.
# In this case, we can construct and update a translation table based on the
# data returned by |deduplicate|.
# It consists of the (potentially) misspelled category names as indices and the
# (potentially) correct categories as values.

# create a table that maps original -> corrected categories
translation_table = pd.Series(deduplicated_data, index=duplicated_names)

# remove duplicates in the original data
translation_table = translation_table[~translation_table.index.duplicated(keep="first")]

###############################################################################
# Since the number of correct spellings will likely be much smaller than the
# number of original categories, we can print the estimated cluster and their
# most common exemplars (the guessed correct spelling):


def print_corrections(spell_correct):
    correct = np.unique(spell_correct.values)
    for c in correct:
        print(
            f"Guessed correct spelling: {c!r} for "
            f"{spell_correct[spell_correct==c].index.values}"
        )


print_corrections(translation_table)

###############################################################################
# In case we want to adapt the translation table post-hoc we can easily
# modify and apply it manually, for instance modifying the
# correspondance for the last entry as such:

translation_table.iloc[-1] = "Completely new category"
new_deduplicated_names = translation_table[duplicated_names]
assert (new_deduplicated_names == "Completely new category").sum() > 0

###############################################################################
# Conclusion
# ----------
#
# In this example, we have seen how to use the |deduplicate| function to
# automatically detect and correct misspelled category names.
# This technique is a different paradigm from the encoding methods implemented
# in the library, at they will instead encode the similarity between entities
# within the dirty categorical feature.
#
# Deduplication is especially useful when we either know our ground truth
# (e.g. the original medication names), or when we know the (dis)similarity
# doesn't provide any useful information for our machine learning task.
