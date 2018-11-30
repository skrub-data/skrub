"""
Scalability considerations for similarity encoding
===================================================

Here we discuss how to apply efficiently SimilarityEncoder to larger
datasets: reducing the number of reference categories to "prototypes",
either chosen as the most frequent categories, or with kmeans clustering.

"""
# Avoid the warning in scikit-learn's LogisticRegression for the change
# in the solver
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

################################################################################
# A tool to report memory usage and run time
# -------------------------------------------
#
# For this example, we build a small tool that reports memory
# usage and compute time of a function
from time import time
import functools
import memory_profiler


def resource_used(func):
    """ Decorator that return a function that prints its usage
    """

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        t0 = time()
        mem, out = memory_profiler.memory_usage((func, args, kwargs),
                                                max_usage=True,
                                                retval=True)
        print("Run time: %.1is    Memory used: %iMb"
              % (time() - t0, mem[0]))
        return out

    return wrapped_func


################################################################################
# Data Importing and preprocessing
# --------------------------------
#
# We first download the dataset:
from dirty_cat.datasets import fetch_traffic_violations

data = fetch_traffic_violations()
print(data['description'])

################################################################################
# Then we load it:
import pandas as pd

df = pd.read_csv(data['path'])

# Limit to 50 000 rows, for a faster example
df = df[:50000].copy()
df = df.dropna(axis=0)
df = df.reset_index()
################################################################################
# We will use SimilarityEncoder on the 'description' column. One
# difficulty is that it many different entries
print(df['Description'].nunique())

################################################################################
print(df['Description'].value_counts()[:20])

################################################################################
# As we will see,SimilarityEncoder takes a while on such data


################################################################################
# SimilarityEncoder with default options
# --------------------------------------
#
# Let us build our vectorizer, using a ColumnTransformer to combine
# one-hot encoding and similarity encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from dirty_cat import SimilarityEncoder

sim_enc = SimilarityEncoder(similarity='ngram')

y = df['Violation Type']

# clean columns
transformers = [('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore'),
                 ['Alcohol',
                  'Arrest Type',
                  'Belts',
                  'Commercial License',
                  'Commercial Vehicle',
                  'Fatal',
                  'Gender',
                  'HAZMAT',
                  'Property Damage',
                  'Race',
                  'Work Zone']),
                ('pass', 'passthrough', ['Year']),
                ]

column_trans = ColumnTransformer(
    # adding the dirty column
    transformers=transformers + [('sim_enc', sim_enc, ['Description'])],
    remainder='drop')

t0 = time()
X = column_trans.fit_transform(df)
t1 = time()
print('Time to vectorize: %s' % (t1 - t0))

################################################################################
# We can run a cross-validation
from sklearn import linear_model, pipeline, model_selection

log_reg = linear_model.LogisticRegression()

model = pipeline.make_pipeline(column_trans, log_reg)
results = resource_used(model_selection.cross_validate)(model, df, y, )
print("Cross-validation score: %s" % results['test_score'])

################################################################################
# Store results for later
scores = dict()
scores['Default options'] = results['test_score']
times = dict()
times['Default options'] = results['fit_time']

################################################################################
# Most frequent strategy to define prototypes
# ---------------------------------------------
#
# The most frequent strategy selects the n most frequent values in a dirty
# categorical variable to reduce the dimensionality of the problem and thus
# speed things up. We select manually the number of prototypes we want to use.
sim_enc = SimilarityEncoder(similarity='ngram', categories='most_frequent',
                            n_prototypes=100)

column_trans = ColumnTransformer(
    # adding the dirty column
    transformers=transformers + [('sim_enc', sim_enc, ['Description'])],
    remainder='drop')

################################################################################
# Check now that prediction is still as good
model = pipeline.make_pipeline(column_trans, log_reg)
results = resource_used(model_selection.cross_validate)(model, df, y)
print("Cross-validation score: %s" % results['test_score'])

################################################################################
# Store results for later
scores['Most frequent'] = results['test_score']
times['Most frequent'] = results['fit_time']

################################################################################
# KMeans strategy to define prototypes
# ---------------------------------------
#
# K-means strategy is also a dimensionality reduction technique.
# SimilarityEncoder can apply a K-means and nearest neighbors algorithm
# to find the prototypes. The number of prototypes is set manually.
sim_enc = SimilarityEncoder(similarity='ngram', categories='k-means',
                            n_prototypes=100)

column_trans = ColumnTransformer(
    # adding the dirty column
    transformers=transformers + [('sim_enc', sim_enc, ['Description'])],
    remainder='drop')

################################################################################
# Check now that prediction is still as good
model = pipeline.make_pipeline(column_trans, log_reg)
results = resource_used(model_selection.cross_validate)(model, df, y)
print("Cross-validation score: %s" % results['test_score'])

################################################################################
# Store results for later
scores['KMeans'] = results['test_score']
times['KMeans'] = results['fit_time']

################################################################################
# Plot a summary figure
# ----------------------
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

################################################################################
# Reduce memory usage during encoding using float32
# ----------------------------------------------------------------
#
# We use a float32 dtype in this example to show some speed and memory gains.
# The use of the scikit-learn model may upcast to float64 (depending on the used
# algorithm). The memory savings will then happen during the encoding.
import numpy as np

sim_enc = SimilarityEncoder(similarity='ngram', dtype=np.float32,
                            categories='most_frequent', n_prototypes=100)

y = df['Violation Type']
# cast the year column to float32
df['Year'] = df['Year'].astype(np.float32)
# clean columns
transformers = [('one_hot', OneHotEncoder(sparse=False, dtype=np.float32,
                                          handle_unknown='ignore'),
                 ['Alcohol',
                  'Arrest Type',
                  'Belts',
                  'Commercial License',
                  'Commercial Vehicle',
                  'Fatal',
                  'Gender',
                  'HAZMAT',
                  'Property Damage',
                  'Race',
                  'Work Zone']),
                ('pass', 'passthrough', ['Year']),
                ]

column_trans = ColumnTransformer(
    # adding the dirty column
    transformers=transformers + [('sim_enc', sim_enc, ['Description'])],
    remainder='drop')

t0 = time()
X = column_trans.fit_transform(df)
t1 = time()
print('Time to vectorize: %s' % (t1 - t0))

################################################################################
# We can run a cross-validation to confirm the memory footprint reduction
model = pipeline.make_pipeline(column_trans, log_reg)
results = resource_used(model_selection.cross_validate)(model, df, y, )
print("Cross-validation score: %s" % results['test_score'])
