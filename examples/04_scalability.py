"""
Scalability considerations for  similarity encoding
===================================================

"""

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

sim_enc =  SimilarityEncoder(similarity='ngram', handle_unknown='ignore')


y = df['Violation Type']

# clean columns
transformers = [('one_hot', OneHotEncoder(sparse=False),
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

from time import time
t0 = time()
X = column_trans.fit_transform(df)
t1 = time()
print('Time to vectorize: %s' % (t1 - t0))
################################################################################
# We can run a cross-validation
from sklearn import linear_model, pipeline, model_selection
model = pipeline.Pipeline([('column_trans', column_trans),
                          ('logistic', linear_model.LogisticRegression())])
t0 = time()
print(model_selection.cross_val_score(model, df, y))
t1 = time()
print('Cross-validation time: %s' % (t1 - t0))

################################################################################
# SimilarityEncoder with hashing
# -------------------------------
#
# The hashing trick can be used to speed up the computation for the ngram
# similarity. This is an approximation, is the sense that collisions can
# occur. The larger the hashing-dim, the less the chances of collisions,
# hence the tighter the approximation.

sim_enc =  SimilarityEncoder(similarity='ngram', handle_unknown='ignore',
                             hashing_dim=2**4)

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
model = pipeline.Pipeline([('column_trans', column_trans),
                           ('logistic', linear_model.LogisticRegression())])
t0 = time()
print(model_selection.cross_val_score(model, df, y))
t1 = time()
print('Cross-validation time: %s' % (t1 - t0))

