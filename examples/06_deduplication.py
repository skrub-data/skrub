"""
Deduplicating misspelled categories with deduplicate
====================================================

A common step in data analyses is grouping or analyzing data conditional on a
categorical variable. In real world datasets, there often will be slight
misspellings in the category names: This happens when, for example, data input
*should* use a drop down menu, but users are forced to input the category name
by hand. Misspellings happen and analyzing the resulting data using a simple
`GROUP BY` is not possible anymore.

This problem is however the perfect use case of *unsupervised learning*, a
category of various statical methods that find structure in data without
providing explicit labels/categories of the data a-priori. Specifically
clustering of the distance between strings can be used to find clusters
of strings that are similar to each other (e.g. differ only by a misspelling)
and hence gives us an easy tool to flag potentially misspelled category names
in an unsupervised manner.
"""

###############################################################################
# An example
# ----------
#
# Imagine the following example:
# As a data scientist, our job is to analyze the data from a hospital ward.
# We notice that most of the cases involve the prescription of one of three different medications:
#  "Contrivan", "Genericon", or "Zipholan".
# However, data entry is manual and - either because the prescribing doctor's handwriting
# was hard to decipher, or due to mistakes during data input - there are multiple
# spelling mistakes for these three medications.
#
# Let's generate some example data that demonstrate this.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_example_data(examples, entries_per_example, prob_mistake_per_letter):
    """Helper function to generate data consisting of multiple entries per example.
    Characters are misspelled with probability `prob_mistake_per_letter`"""
    import string

    data = []
    for example, n_ex in zip(examples, entries_per_example):
        len_ex = len(example)
        # generate a 2D array of chars of size (n_ex, len_ex)
        str_as_list = np.array([list(example)] * n_ex)
        # randomly choose which characters are misspelled
        idxes = np.where(
            np.random.random(len(example[0]) * n_ex) < prob_mistake_per_letter
        )[0]
        # and randomly pick with which character to replace
        replacements = [
            string.ascii_lowercase[i]
            for i in np.random.choice(np.arange(26), len(idxes)).astype(int)
        ]
        # introduce spelling mistakes at right examples and right char locations per example
        str_as_list[idxes // len_ex, idxes % len_ex] = replacements
        # go back to 1d array of strings
        data.append(np.ascontiguousarray(str_as_list).view(f"U{len_ex}").ravel())
    return np.concatenate(data)


# set seed for reproducibility
np.random.seed(123)

# our three medication names
medications = ["Contrivan", "Genericon", "Zipholan"]
entries_per_medications = [500, 100, 1500]

# 5% probability of a typo per letter
prob_mistake_per_letter = 0.05

data = generate_example_data(
    medications, entries_per_medications, prob_mistake_per_letter
)
# we extract the unique medication names in the data & how often they appear
unique_examples, counts = np.unique(data, return_counts=True)
# and build a series out of them
ex_series = pd.Series(counts, index=unique_examples)

###############################################################################
# Visualize the data
# ------------------

ex_series.plot.barh(figsize=(10, 15))
plt.xlabel("Medication name")
plt.ylabel("Counts")

###############################################################################
# We can now see clearly the structure of the data: The three original medications
# are the most common ones, however there are many spelling mistakes and hence
# many slight variations of the names of the original medications.
#
# The idea is to use the fact that the string-distance of each misspelled medication
# name will be closest to either the correctly or incorrectly spelled orginal
# medication name - and therefore form clusters.


from dirty_cat import deduplicate
from dirty_cat._deduplicate import _compute_ngram_distance
from scipy.spatial.distance import squareform

ngram_distances = _compute_ngram_distance(unique_examples)
square_distances = squareform(ngram_distances)

###############################################################################
# We can visualize the pair-wise distance between all medication names
# --------------------------------------------------------------------
#
# Below we use a heatmap to visualize the pairwise-distance between medication names.
# A darker color means that two medication names are closer together (i.e. more similar),
# a lighter color means a larger distance. We can see that we are dealing with three
# clusters - the original medication names and their misspellings that cluster around them.

import seaborn as sns

fig, axes = plt.subplots(1, 1, figsize=(12, 12))
sns.heatmap(
    square_distances, yticklabels=ex_series.index, xticklabels=ex_series.index, ax=axes
)

###############################################################################
# Clustering to suggest corrections of misspelled names
# -----------------------------------------------------
#
# The number of clusters will need some adjustment depending on the data you have.
# If no fixed number of clusters is given, `deduplicate` tries to set it automatically
# via the `silhouette score <https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient>`_.


deduplicated_data = deduplicate(data)

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
# In this example we can correct all spelling mistakes by using the ideal number
# of clusters as determined by the silhouette score.
#
# However, often the translation/deduplication won't be perfect and will require some tweaks.
# In this case, we can construct and update a translation table based on the data
# returned by `deduplicate`.
# It consists of the (potentially) misspelled category names as indices and the
# (potentially) correct categories as values.

# create a table that maps original -> corrected categories
translation_table = pd.Series(deduplicated_data, index=data)

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
# In case we want to adapt the translation table post-hoc we can easily do so:

translation_table["Gszericon"] = "Completely new category"
new_deduplicated_data = translation_table[data]
assert (new_deduplicated_data == "Completely new category").sum() > 0
