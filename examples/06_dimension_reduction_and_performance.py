"""
Scalability considerations for similarity encoding
===================================================

Here we discuss how to apply efficiently SimilarityEncoder to larger
datasets: reducing the number of reference categories to "prototypes",
either chosen as the most frequent categories, or with kmeans clustering.

Note that the :class:`GapEncoder` naturally does data reduction and comes
with online estimation. As a result is it more scalable than the
SimilarityEncoder, and should be preferred in large-scale settings.

"""
# Avoid the warning in scikit-learn's LogisticRegression for the change
# in the solver
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

###############################################################################
# A tool to report memory usage and run time
# ------------------------------------------
#
# For this example, we build a small tool that reports memory
# usage and compute time of a function
from time import time
import functools
import tracemalloc


def resource_used(func):
    """ Decorator that return a function that prints its usage
    """

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        t0 = time()
        tracemalloc.start()
        out = func(*args, **kwargs)
        size, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak /= (1024 ** 2)  # Convert to megabytes
        print("Run time: %.1is    Memory used: %iMb"
              % (time() - t0, peak))
        return out

    return wrapped_func


###############################################################################
# Data Importing and preprocessing
# --------------------------------
#
# We first get the dataset:
import pandas as pd
from dirty_cat.datasets import fetch_open_payments

open_payments = fetch_open_payments()
print(open_payments.description)

df = open_payments.X

na_mask: pd.DataFrame = df.isna()
df = df.dropna(axis=0)
df = df.reset_index()

from functools import reduce

y = open_payments.y
# Combine boolean masks
na_mask = reduce(lambda acc, col: acc | na_mask[col],
                 na_mask.columns, na_mask[na_mask.columns[0]])
# Drop the lines that contained missing values in X
y = y[~na_mask]
y.reset_index()

clean_columns = [
    'Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name',
    'Dispute_Status_for_Publication',
    'Physician_Specialty',
]
dirty_columns = [
    'Name_of_Associated_Covered_Device_or_Medical_Supply1',
    'Name_of_Associated_Covered_Drug_or_Biological1',
]

###############################################################################
# We will use SimilarityEncoder on the the two dirty columns defined above.
# One difficulty is that they have many different entries.
print(df[dirty_columns].nunique())

###############################################################################
print(df[dirty_columns].value_counts()[:20])

###############################################################################
# As we will see, SimilarityEncoder takes a while on such data.


###############################################################################
# SimilarityEncoder with default options
# --------------------------------------
#
# Let us build our vectorizer, using a ColumnTransformer to combine
# one-hot encoding and similarity encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from dirty_cat import SimilarityEncoder

sim_enc = SimilarityEncoder()

transformers = [
    ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore'), clean_columns),
]

column_trans = ColumnTransformer(
    transformers=transformers + [('sim_enc', sim_enc, dirty_columns)],
    remainder='drop')

t0 = time()
X = column_trans.fit_transform(df)
t1 = time()
print('Time to vectorize: %s' % (t1 - t0))

###############################################################################
# We can run a cross-validation
from sklearn import linear_model, pipeline, model_selection

# We specify max_iter to avoid convergence warnings
log_reg = linear_model.LogisticRegression(max_iter=10000)

model = pipeline.make_pipeline(column_trans, log_reg)
results = resource_used(model_selection.cross_validate)(model, df, y)
print("Cross-validation score: %s" % results['test_score'])

###############################################################################
# Store results for later
scores = dict()
scores['Default options'] = results['test_score']
times = dict()
times['Default options'] = results['fit_time']

###############################################################################
# Most frequent strategy to define prototypes
# -------------------------------------------
#
# The most frequent strategy selects the n most frequent values in a dirty
# categorical variable to reduce the dimensionality of the problem and thus
# speed things up. We select manually the number of prototypes we want to use.
sim_enc = SimilarityEncoder(categories='most_frequent', n_prototypes=100)

column_trans = ColumnTransformer(
    transformers=transformers + [('sim_enc', sim_enc, dirty_columns)],
    remainder='drop')

###############################################################################
# Check now that prediction is still as good
model = pipeline.make_pipeline(column_trans, log_reg)
results = resource_used(model_selection.cross_validate)(model, df, y)
print("Cross-validation score: %s" % results['test_score'])

###############################################################################
# Store results for later
scores['Most frequent'] = results['test_score']
times['Most frequent'] = results['fit_time']

###############################################################################
# KMeans strategy to define prototypes
# ------------------------------------
#
# K-means strategy is also a dimensionality reduction technique.
# SimilarityEncoder can apply a K-means and nearest neighbors algorithm
# to find the prototypes. The number of prototypes is set manually.
sim_enc = SimilarityEncoder(categories='k-means', n_prototypes=100)

column_trans = ColumnTransformer(
    transformers=transformers + [('sim_enc', sim_enc, dirty_columns)],
    remainder='drop')

###############################################################################
# Check now that prediction is still as good
model = pipeline.make_pipeline(column_trans, log_reg)
results = resource_used(model_selection.cross_validate)(model, df, y)
print("Cross-validation score: %s" % results['test_score'])

###############################################################################
# Store results for later
scores['KMeans'] = results['test_score']
times['KMeans'] = results['fit_time']

###############################################################################
# Plot a summary figure
# ---------------------
import seaborn
import matplotlib.pyplot as plt

_, (ax1, ax2) = plt.subplots(nrows=2, figsize=(4, 3))
seaborn.boxplot(data=pd.DataFrame(scores), orient='h', ax=ax1)
ax1.set_xlabel('Prediction accuracy', size=16)
[t.set(size=16) for t in ax1.get_yticklabels()]

seaborn.boxplot(data=pd.DataFrame(times), orient='h', ax=ax2)
ax2.set_xlabel('Computation time', size=16)
[t.set(size=16) for t in ax2.get_yticklabels()]
plt.tight_layout()
