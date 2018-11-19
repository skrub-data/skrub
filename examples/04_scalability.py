"""
Scalability considerations for  similarity encoding
===================================================

"""
import warnings

################################################################################
# A tool to report memory usage and run time of a function
# ---------------------------------------------------------
#
# For the sake of this example, we build a small tool that reports memory
# usage and compute time of a function
from time import time
import functools
import memory_profiler

warnings.simplefilter(action='ignore', category=FutureWarning)


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
print(df['Description'].value_counts()[:30])

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

print('\nBasic similarity encoding')

sim_enc = SimilarityEncoder(similarity='ngram', handle_unknown='ignore')

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
log_ref = linear_model.LogisticRegression()

model = pipeline.make_pipeline(column_trans, log_ref)
print("Cross-validation score: %s" %
      resource_used(model_selection.cross_val_score)(model, df, y))


################################################################################
# SimilarityEncoder with most frequent strategy
# ---------------------------------------------
#
# The most frequent strategy selects the n most frequent values in a dirty
# categorical variable to reduce the dimensionality of the problem and thus
# speed things up. We select manually the number of prototypes we want to use.

print('\nSimilarity encoding with a most frequent strategy')

sim_enc = SimilarityEncoder(similarity='ngram', categories='most_frequent', n_prototypes=100)

column_trans = ColumnTransformer(
    # adding the dirty column
    transformers=transformers + [('sim_enc', sim_enc, ['Description'])],
    remainder='drop')

from time import time

t0 = time()
X = column_trans.fit_transform(df)
t1 = time()
print('Time to vectorize: %s' % (t1 - t0))

################################################################################
# Check now that prediction is still as good
model = pipeline.make_pipeline(column_trans, log_ref)
print("Cross-validation score: %s" %
      resource_used(model_selection.cross_val_score)(model, df, y))


################################################################################
# SimilarityEncoder with k-means strategy
# ---------------------------------------
#
# K-means strategy is also a dimensionality reduction technique. But we apply
# a K-means and nearest neighbors algorithm to find the prototypes. The number
# of prototypes is set manually.

print('\nSimilarity encoding with a k-means strategy')

sim_enc = SimilarityEncoder(similarity='ngram', categories='k-means', n_prototypes=100)

column_trans = ColumnTransformer(
    # adding the dirty column
    transformers=transformers + [('sim_enc', sim_enc, ['Description'])],
    remainder='drop')

from time import time

t0 = time()
X = column_trans.fit_transform(df)
t1 = time()
print('Time to vectorize: %s' % (t1 - t0))

################################################################################
# Check now that prediction is still as good
model = pipeline.make_pipeline(column_trans, log_ref)
print("cross-validation score: %s" %
      resource_used(model_selection.cross_val_score)(model, df, y))
