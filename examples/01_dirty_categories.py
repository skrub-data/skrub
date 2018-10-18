"""
Investigating dirty categories
=================================

What are dirty categorical variables and how can a good encoding help
with statistical learning.
"""

#########################################################################
# What do we mean by dirty categories?
# -------------------------------------------------
#
# Let's look at a dataset called employee salaries:
import pandas as pd
from dirty_cat import datasets

employee_salaries = datasets.fetch_employee_salaries()
print(employee_salaries['description'])
data = pd.read_csv(employee_salaries['path'])
print(data.head(n=5))

#########################################################################
# Here is how many unique entries there is per column
print(data.nunique())

#########################################################################
# As we can see, some entries have many different unique values:
print(data['Employee Position Title'].value_counts().sort_index())

#########################################################################
# These different entries are often variations on the same entities:
# there are 3 kinds of Accountant/Auditor.
#
# Such variations will break traditional categorical encoding methods:
#
# * Using simple one-hot encoding will create orthogonal features,
#   whereas it is clear that those 3 terms have a lot in common.
#
# * If we wanted to use word embedding methods such as word2vec,
#   we would have to go through a cleaning phase: those algorithms
#   are not trained to work on data such as 'Accountant/Auditor I'.
#   However, this can be error prone and time consuming.
#
# The problem becomes easier if we can capture relationships between
# entries.
#
# To simplify understanding, we will focus on the column describing the
# employee's position title:
# data
values = data[['Employee Position Title', 'Gender', 'Current Annual Salary']]

#########################################################################
# String similarity between entries
# -------------------------------------------------
#
# That's where our encoders get into play. In order to robustly
# embed dirty semantic data, the SimilarityEncoder creates a similarity
# matrix based on the 3-gram structure of the data.
sorted_values = values['Employee Position Title'].sort_values().unique()

from dirty_cat import SimilarityEncoder

similarity_encoder = SimilarityEncoder(similarity='ngram')
transformed_values = similarity_encoder.fit_transform(
    sorted_values.reshape(-1, 1))

#########################################################################
# Plotting the new representation using multi-dimensional scaling
# ................................................................
#
# Let's now plot a couple points at random using a low-dimensional representation
# to get an intuition of what the similarity encoder is doing:
from sklearn.manifold import MDS

mds = MDS(dissimilarity='precomputed', n_init=10, random_state=42)
two_dim_data = mds.fit_transform(
    1 - transformed_values)  # transformed values lie
# in the 0-1 range, so 1-transformed_value yields a positive dissimilarity matrix
print(two_dim_data.shape)
print(sorted_values.shape)

#########################################################################
# We first quickly fit a KNN so that the plots does not get too busy:
import numpy as np

n_points = 5
np.random.seed(42)
from sklearn.neighbors import NearestNeighbors

random_points = np.random.choice(len(similarity_encoder.categories_[0]),
                                 n_points, replace=False)
nn = NearestNeighbors(n_neighbors=2).fit(transformed_values)
_, indices_ = nn.kneighbors(transformed_values[random_points])
indices = np.unique(indices_.squeeze())

#########################################################################
# Then we plot it, adding the categories in the scatter plot:

import matplotlib.pyplot as plt

f, ax = plt.subplots()
ax.scatter(x=two_dim_data[indices, 0], y=two_dim_data[indices, 1])
# adding the legend
for x in indices:
    ax.text(x=two_dim_data[x, 0], y=two_dim_data[x, 1], s=sorted_values[x],
            fontsize=8)
ax.set_title(
    'multi-dimensional-scaling representation using a 3gram similarity matrix')

#########################################################################
# Heatmap of the similarity matrix
# ................................
#
# We can also plot the distance matrix for those observations:
f2, ax2 = plt.subplots(figsize=(6, 6))
cax2 = ax2.matshow(transformed_values[indices, :][:, indices])
ax2.set_yticks(np.arange(len(indices)))
ax2.set_xticks(np.arange(len(indices)))
ax2.set_yticklabels(sorted_values[indices], rotation='30')
ax2.set_xticklabels(sorted_values[indices], rotation='60', ha='right')
ax2.xaxis.tick_bottom()
ax2.set_title('Similarities across categories')
f2.colorbar(cax2)
f2.tight_layout()

########################################################################
# As shown in the previous plot, we see that the nearest neighbor of
# "Communication Equipment Technician"
# is "telecommunication technician", although it is also
# very close to senior "supply technician": therefore, we grasp the
# "communication" part (not initially present in the category as a unique word)
# as well as the technician part of this category.


#########################################################################
# Encoding categorical data using SimilarityEncoder
# -------------------------------------------------
#
# A typical data-science workflow uses one-hot encoding to represent
# categories.
from sklearn.preprocessing import OneHotEncoder

# encoding simply a subset of the observations
n_obs = 20
employee_position_titles = values['Employee Position Title'].head(
    n_obs).to_frame()
categorical_encoder = OneHotEncoder(sparse=False)
one_hot_encoded = categorical_encoder.fit_transform(employee_position_titles)
f3, ax3 = plt.subplots(figsize=(6, 6))
ax3.matshow(one_hot_encoded)
ax3.set_title('Employee Position Title values, one-hot encoded')
ax3.axis('off')
f3.tight_layout()

#########################################################################
# The corresponding is very sparse
#
# SimilarityEncoder can be used to replace one-hot encoding capturing the
# similarities:

f4, ax4 = plt.subplots(figsize=(6, 6))
similarity_encoded = similarity_encoder.fit_transform(employee_position_titles)
ax4.matshow(similarity_encoded)
ax4.set_title('Employee Position Title values, similarity encoded')
ax4.axis('off')
f4.tight_layout()

#########################################################################
# Other examples in the dirty_cat documentation show how
# similarity encoding impacts prediction performance.
