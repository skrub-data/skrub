"""
Feature interpretation with the GapEncoder
==========================================

We illustrate here how categorical encodings obtained with the GapEncoder
can be interpreted in terms of latent topics. We use as example the
`employee salaries <https://catalog.data.gov/dataset/employee-salaries-2016>`_
dataset, and encode the column *Employee Position Title*, that contains dirty
categorical data.

"""

################################################################################
# Data Importing
# --------------
#
# We first download the dataset:
from dirty_cat.datasets import fetch_employee_salaries
employee_salaries = fetch_employee_salaries()
print(employee_salaries['DESCR'])


################################################################################
# Then we load it:
import pandas as pd
df = employee_salaries['data']

################################################################################
# Now, we retrieve the dirty column to encode:
dirty_column = 'employee_position_title'
X_dirty = df[dirty_column]
print(X_dirty.head(), end='\n\n')
print(f'Number of dirty entries = {len(X_dirty)}')

################################################################################
# Encoding dirty job titles
# -------------------------
#
# We first create an instance of the GapEncoder with n_components=10:
from dirty_cat import GapEncoder
enc = GapEncoder(n_components=10, random_state=42)

################################################################################
# Then we fit the model on the dirty categorical data and transform it to
# obtain encoded vectors of size 10:
X_enc = enc.fit_transform(X_dirty)
print(f'Shape of encoded vectors = {X_enc.shape}')

################################################################################
# Interpreting encoded vectors
# ----------------------------
#
# The GapEncoder can be understood as a continuous encoding on a set of latent
# topics estimated from the data. The latent topics are built by
# capturing combinations of substrings that frequently co-occur, and encoded
# vectors correspond to their activations.
# To interpret these latent topics, we select for each of them a few labels
# from the input data with the highest activations.
# In the example below we select 3 labels to summarize each topic.

topic_labels = enc.get_feature_names(n_labels=3)
for k in range(len(topic_labels)):
    labels = topic_labels[k]
    print(f'Topic n°{k}: {labels}') 
    
################################################################################
# As expected, topics capture labels that frequently co-occur. For instance,
# the labels *firefighter*, *rescuer*, *rescue* appear together in
# *Firefigther/Rescuer III*, or *Fire/Rescue Lieutenant*. We can qualitatively
# check that these labels "summarize" well each topic by looking at their
# encoded vectors:
import matplotlib.pyplot as plt
encoded_labels = enc.transform(topic_labels)
encoded_labels /= encoded_labels.sum(axis=1)
plt.figure(figsize=(8,4))
plt.imshow(encoded_labels)
plt.xlabel('Latent topics', size=12)
plt.xticks(range(0, 10))
plt.ylabel('Topic labels', size=12)
plt.yticks(range(0, 10), labels=topic_labels)
plt.colorbar(ticks=[0, 0.5, 1]).set_label(label='Topic activations', size=12)
plt.clim(-0.05, 1.05)
plt.tight_layout()
plt.show()

################################################################################
# As we can see, the encoded topic labels have very sparse activations: one set
# of labels correspond to one latent topic only. They can thus be reliably
# used to summarize each topic.