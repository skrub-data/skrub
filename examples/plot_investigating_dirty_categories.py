"""
Basic dirty_cat example: manipulating and looking at data
=========================================================

let's try to understand how embedding dirty categorical variables with
3gram similarity can help in learning better models
"""

#########################################################################
# What do we mean by dirty data?
# -------------------------------------------------
#
# Let's look at a dataset called employee salaries:
import pandas as pd
from dirty_cat import datasets

employee_salaries = datasets.fetch_employee_salaries()
df = pd.read_csv(employee_salaries['path'])
print(df.head(n=5))

#########################################################################
# Here is how the columns are distributed:
print(df.nunique())

#########################################################################
# Some numerical columns (Gross pay, etc..) and some obvious categorical
# columns such as full_name
# of course have many different values. but it is also the case
# for other categorical columns  such as Employee position title:

sorted_values = df['Employee Position Title'].sort_values().unique()
for i in range(5):
    print(sorted_values[i] + '\n')

#########################################################################
# Here we go! See how there are 3 kinds of Accountant/Auditor? I,II,and III.
# Now, there are some reason why traditional word-encoding methods won't work
# very well.
#
# * Using simple one-hot encoding will create orthogonal features,
#   whereas it is clear that those 3 terms have a lot in common.
#
# * If we wanted to use word embedding methods such as word2vec,
#   we would have to go through a cleaning phase: those algorithms
#   are not trained to work on data such as 'Accountant/Auditor I'.
#   However, that can be unsafe and take a long time.


#########################################################################
# Encoding categorical data using SimilarityEncoder
# -------------------------------------------------
# That's where our encoders get into play. In order to robustly
# embed dirty semantic data, the SimilarityEncoder creates a similarity
# matrix based on the 3-gram structure of the data.
from dirty_cat import SimilarityEncoder

similarity_encoder = SimilarityEncoder(similarity='ngram')
transformed_values = similarity_encoder.fit_transform(
    sorted_values.reshape(-1, 1))

#########################################################################
# ------------------------------------------------------------
# Plotting the new feature map using multi-dimensional scaling
# ------------------------------------------------------------
# lets now plot a couple points at random using a low-dimensional representation
# to get an intuition of what the similarity encoder is doing:
from sklearn.manifold import MDS

mds = MDS(dissimilarity='precomputed', n_init=10, random_state=42)
two_dim_data = mds.fit_transform(
    1 - transformed_values)  # transformed values lie
# in the 0-1 range, so 1-transformed_value yields a positive dissimilarity matrix
print(two_dim_data.shape)
print(sorted_values.shape)

#########################################################################
# we first quickly fit a KNN so that the plots does not get too busy:
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
# and then plot it, adding the categories in the scatter plot:

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
# ------------------------------------------------------------
# Heatmap of the similarity matrix
# ------------------------------------------------------------
# We can also plot the distance matrix for those observations:
f2, ax2 = plt.subplots(figsize=(6, 6))
cax2 = ax2.matshow(transformed_values[indices, :][:, indices])
ax2.set_yticks(np.arange(len(indices)))
ax2.set_xticks(np.arange(len(indices)))
ax2.set_yticklabels(sorted_values[indices], rotation='30')
ax2.set_xticklabels(sorted_values[indices], rotation='60', ha='right')
ax2.xaxis.tick_bottom()
f2.colorbar(cax2)
f2.tight_layout()

########################################################################
# As shown in the previous plot, we see that "communication Equipment technician"'s
# nearest neighbor is "telecommunication technician", although it is also
# very close to senior "supply technician": therefore, we grasp the
# "communication" part (not initially present in the category as a unique word)
# as well as the technician part of this category.


