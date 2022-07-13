"""
Scalability considerations for similarity encoding
==================================================

We discuss in this notebook how to efficiently apply the |SE| to larger
datasets: reducing the number of reference categories to "prototypes",
either chosen as the most frequent categories, or with kmeans clustering.


.. note::
    The |Gap| naturally does data reduction and comes with online estimation.
    As a result, is it more scalable than the |SE|,
    and should be preferred in large-scale settings.


.. |SE| replace:: :class:`~dirty_cat.SimilarityEncoder`

.. |Gap| replace:: :class:`~dirty_cat.GapEncoder`

.. |ColumnTransformer| replace:: :class:`~sklearn.compose.ColumnTransformer`

.. |OHE| replace:: :class:`~sklearn.preprocessing.OneHotEncoder`

"""
# Avoid the warning in scikit-learn's LogisticRegression for the change
# in the solver
# TODO: move this to the exact place where the warnings are raised
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)

###############################################################################
# A tool to report memory usage and run time
# ------------------------------------------
#
# For this example, we build a small tool that reports memory
# usage and compute time of a function
from time import perf_counter
import functools
import tracemalloc


def resource_used(func):
    """
    Decorator for performance analysis.
    """

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        t0 = perf_counter()  # Launch a time
        tracemalloc.start()
        out = func(*args, **kwargs)
        size, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak /= (1024 ** 2)  # Convert to megabytes
        print(f"Run time: {perf_counter() - t0:.2f}s ; "
              f"Memory used: {peak:.2f}MB. ")
        return out

    return wrapped_func


###############################################################################
# Data Importing and preprocessing
# --------------------------------
#
# First, let's fetch the dataset we'll use further down
import pandas as pd
from dirty_cat.datasets import fetch_open_payments

open_payments = fetch_open_payments()
X = open_payments.X

open_payments.description

###############################################################################
# We'll perform a some cleaning
from functools import reduce

# Remove the missing lines in X
na_mask: pd.DataFrame = X.isna()
X = X.dropna(axis=0).reset_index(drop=True)

y = open_payments.y
# Combine boolean masks ; TODO: simplify
na_mask = reduce(lambda acc, col: acc | na_mask[col],
                 na_mask.columns, na_mask[na_mask.columns[0]])
# Drop the lines in y that contained missing values in X
y = y[~na_mask].reset_index(drop=True)

###############################################################################
# We'll write down which columns are clean and which are dirty
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
# We will use |SE| on the two dirty columns defined above.
# One difficulty is that they have many entries, and because of that, as we'll
# see, the :code:`SimilarityEncoder` will take a while.
X[dirty_columns].value_counts()[:20]

###############################################################################
X[dirty_columns].nunique()

###############################################################################
# SimilarityEncoder with default options
# --------------------------------------
#
# Let us build our vectorizer, using a |ColumnTransformer| to combine
# a |OHE| and a |SE|
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from dirty_cat import SimilarityEncoder

clean_col_transformer = [
    ('one_hot',
     OneHotEncoder(sparse=False, handle_unknown='ignore'),
     clean_columns),
]

column_trans = ColumnTransformer(
    transformers=clean_col_transformer + [
        ('sim_enc',
         SimilarityEncoder(similarity='ngram'),
         dirty_columns)
    ],
    remainder='drop')

t0 = perf_counter()
X_enc = column_trans.fit_transform(X)
t1 = perf_counter()
print(f'Time to vectorize: {t1 - t0:.3f}s')

###############################################################################
# Let's now run a cross-validation !
from sklearn import pipeline, model_selection
from sklearn.linear_model import LogisticRegression

# We specify max_iter to avoid convergence warnings
log_reg = LogisticRegression(max_iter=10000)

model = pipeline.make_pipeline(column_trans, log_reg)
results = resource_used(model_selection.cross_validate)(model, X, y)
print(f"Cross-validation score: {results['test_score']}")

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
# The :code:`most_frequent` strategy selects the :code:`n` most frequent
# values in a dirty categorical variable to reduce the dimensionality of the
# problem and thus speed things up.
# Here, we arbitrarily choose 100 as the number of prototypes we want to use.

column_trans = ColumnTransformer(
    transformers=clean_col_transformer + [
        ('sim_enc',
         SimilarityEncoder(similarity='ngram', categories='most_frequent',
                           n_prototypes=100),
         dirty_columns)
    ],
    remainder='drop')

###############################################################################
# Check that the prediction is still as good
model = pipeline.make_pipeline(column_trans, log_reg)
results = resource_used(model_selection.cross_validate)(model, X, y)
print(f"Cross-validation score: {results['test_score']}")

###############################################################################
# Store results for later
scores['Most frequent'] = results['test_score']
times['Most frequent'] = results['fit_time']

###############################################################################
# KMeans strategy to define prototypes
# ------------------------------------
#
# The k-means strategy is also a dimensionality reduction technique.
# The :code:`SimilarityEncoder` can apply a K-means and nearest neighbors
# algorithm to find the prototypes. Once again, the number of prototypes
# we chose here is arbitrary.

column_trans = ColumnTransformer(
    transformers=clean_col_transformer + [
        ('sim_enc',
         SimilarityEncoder(similarity='ngram', categories='k-means',
                           n_prototypes=100),
         dirty_columns)
    ],
    remainder='drop')

###############################################################################
# Check that the prediction is still as good
model = pipeline.make_pipeline(column_trans, log_reg)
results = resource_used(model_selection.cross_validate)(model, X, y)
print("Cross-validation score: %s" % results['test_score'])

###############################################################################
# Store results for later
scores['KMeans'] = results['test_score']
times['KMeans'] = results['fit_time']

###############################################################################
# Summary
# -------
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
