"""
.. _example_explaining_similarity_encoding:

===================================
Investigating the SimilarityEncoder
===================================

In this example, we will take a deeper look at how the |SimilarityEncoder|
works, and how it compares to the |OneHotEncoder| and other encoders in the
library.


.. |OneHotEncoder| replace::
    :class:`~sklearn.preprocessing.OneHotEncoder`

.. |SimilarityEncoder| replace::
    :class:`~skrub.SimilarityEncoder`
"""

###############################################################################
# What do we mean by dirty categories?
# ------------------------------------
#
# Let's get the table we will use in this example,
# the `employee salaries dataset <https://www.openml.org/d/42125>`_:

from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
print(dataset.description)

X = dataset.X
y = dataset.y

X.head()


###############################################################################
# These different entries are often variations of the same entity.
# For example, there are 3 kinds of "Accountant/Auditor".
#
# Such variations will break traditional categorical encoding methods:
#
# * Using a simple |OneHotEncoder|
#   will create orthogonal features, whereas it is clear that
#   those 3 terms have a lot in common.
#
# * If we wanted to use word embedding methods such as
#   `Word2vec <https://www.tensorflow.org/tutorials/text/word2vec>`_,
#   we would have to go through a cleaning phase: those algorithms
#   are not trained to work on data such as "Accountant/Auditor I".
#   However, this can be error-prone and time-consuming.
#
# The problem becomes easier if we can capture relationships between
# entries.
#
# To simplify understanding, we will focus on the column describing the
# employee's position title:

values = X[["employee_position_title", "gender"]]
values.insert(0, "current_annual_salary", dataset.y)

sorted_values = values["employee_position_title"].sort_values().unique()

###############################################################################
# In order to robustly embed dirty semantic data, the |SimilarityEncoder|
# creates a similarity matrix based on an n-gram representation of the data.

from skrub import SimilarityEncoder

similarity_encoder = SimilarityEncoder()
transformed_values = similarity_encoder.fit_transform(sorted_values.reshape(-1, 1))

###############################################################################
# Plotting the new representation using multi-dimensional scaling
# ---------------------------------------------------------------
#
# Let's now plot a couple of points at random using a low-dimensional
# representation to get an intuition of what the |SimilarityEncoder| is doing:

from sklearn.manifold import MDS

mds = MDS(dissimilarity="precomputed", n_init=10, random_state=42)
two_dim_data = mds.fit_transform(1 - transformed_values)
# transformed values lie in the 0-1 range,
# so 1-transformed_value yields a positive dissimilarity matrix
print(two_dim_data.shape)
print(sorted_values.shape)

###############################################################################
# We first fit a KNN so that the plots does not get too busy:

import numpy as np
from sklearn.neighbors import NearestNeighbors

n_points = 5
np.random.seed(42)

random_points = np.random.choice(
    len(similarity_encoder.categories_[0]), n_points, replace=False
)
nn = NearestNeighbors(n_neighbors=2).fit(transformed_values)
_, indices_ = nn.kneighbors(transformed_values[random_points])
indices = np.unique(indices_.squeeze())

###############################################################################
# Then we plot it, adding the categories in the scatter plot:

import matplotlib.pyplot as plt

f, ax = plt.subplots()
ax.scatter(x=two_dim_data[indices, 0], y=two_dim_data[indices, 1])
# adding the legend
for x in indices:
    ax.text(
        x=two_dim_data[x, 0],
        y=two_dim_data[x, 1],
        s=sorted_values[x],
        fontsize=8,
    )
ax.set_title(
    "multi-dimensional-scaling representation using a 3-gram similarity matrix"
)

###############################################################################
# Heatmap of the similarity matrix
# --------------------------------
#
# We can also plot the distance matrix for those observations:

f2, ax2 = plt.subplots(figsize=(6, 6))
cax2 = ax2.matshow(transformed_values[indices, :][:, indices])
ax2.set_yticks(np.arange(len(indices)))
ax2.set_xticks(np.arange(len(indices)))
ax2.set_yticklabels(sorted_values[indices], rotation=30)
ax2.set_xticklabels(sorted_values[indices], rotation=60, ha="right")
ax2.xaxis.tick_bottom()
ax2.set_title("Similarities across categories")
f2.colorbar(cax2)
f2.tight_layout()

###############################################################################
# As shown in the previous plot, we see that the nearest neighbor of
# "Communication Equipment Technician"
# is "Telecommunication Technician", although it is also
# very close to "Senior Supply Technician": therefore, the |SimilarityEncoder| grasps the
# "Communication" part (not initially present in the category as a unique word)
# as well as the "Technician" part of this category.
#
# In conclusion, the |SimilarityEncoder| works as a generalized upgrade of the |OneHotEncoder|, that takes into account string similarities.
