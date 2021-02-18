"""
Semantic variation in the "Midwest"
===================================
Here's some survey data with one dirty column, consisting of
an open-ended question, on which one-hot encoding does not work well.
The other columns are more traditional categorical or numerical
variables.

Let's see how different encoding for the dirty column impact on the
score of a classification problem.

"""

################################################################################
# Loading the data
# ----------------
from dirty_cat.datasets import fetch_midwest_survey
import pandas as pd

dataset = fetch_midwest_survey()
print(dataset['path'])
df = pd.read_csv(dataset['path'], quotechar="'", escapechar='\\')

################################################################################
# The challenge with this data is that it contains a free-form input
# column, where people put whatever they want:
dirty_column = 'What_would_you_call_the_part_of_the_country_you_live_in_now'
print(df[dirty_column].value_counts()[-10:])

################################################################################
# Separating clean, and dirty columns as well a a column we will try to predict
# ------------------------------------------------------------------------------

target_column = 'Census_Region'
clean_columns = [
    'What_would_you_call_the_part_of_the_country_you_live_in_now',
    'Do_you_consider_Illinois_state_as_part_of_the_Midwest',
    'Do_you_consider_Indiana_state_as_part_of_the_Midwest',
    'Do_you_consider_Iowa_state_as_part_of_the_Midwest',
    'Do_you_consider_Kansas_state_as_part_of_the_Midwest',
    'Do_you_consider_Michigan_state_as_part_of_the_Midwest',
    'Do_you_consider_Minnesota_state_as_part_of_the_Midwest',
    'Do_you_consider_Missouri_state_as_part_of_the_Midwest',
    'Do_you_consider_Nebraska_state_as_part_of_the_Midwest',
    'Do_you_consider_North_Dakota_state_as_part_of_the_Midwest',
    'Do_you_consider_Ohio_state_as_part_of_the_Midwest',
    'Do_you_consider_South_Dakota_state_as_part_of_the_Midwest',
    'Do_you_consider_Wisconsin_state_as_part_of_the_Midwest',
    'Do_you_consider_Arkansas_state_as_part_of_the_Midwest',
    'Do_you_consider_Colorado_state_as_part_of_the_Midwest',
    'Do_you_consider_Kentucky_state_as_part_of_the_Midwest',
    'Do_you_consider_Oklahoma_state_as_part_of_the_Midwest',
    'Do_you_consider_Pennsylvania_state_as_part_of_the_Midwest',
    'Do_you_consider_West_Virginia_state_as_part_of_the_Midwest',
    'Do_you_consider_Montana_state_as_part_of_the_Midwest',
    'Do_you_consider_Wyoming_state_as_part_of_the_Midwest',
    'Gender',
    'Age',
    'Household_Income',
    'Education']
y = df[target_column].values.ravel()

##############################################################################
# A pipeline for data fitting and prediction
# -------------------------------------------
# We first import the right encoders to transform our clean/dirty data:
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from dirty_cat import SimilarityEncoder, MinHashEncoder,\
    GapEncoder

encoder_dict = {
    'one-hot': OneHotEncoder(handle_unknown='ignore', sparse=False),
    'similarity': SimilarityEncoder(similarity='ngram'),
    'minhash': MinHashEncoder(),
    'gap': GapEncoder(),
    'num': FunctionTransformer(None)
}
##############################################################################
# All the clean columns are encoded once and for all, but since we
# benchmark different categorical encodings for the dirty variable,
# we create a function that takes an encoding as an input, and returns a \
# scikit-learn pipeline for our problem.
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def make_pipeline(encoding_method):
    # static transformers from the other columns
    transformers = [('one-hot-clean', encoder_dict['one-hot'], clean_columns)]
    # adding the encoded column
    transformers += [(encoding_method + '-dirty', encoder_dict[encoding_method],
                      [dirty_column])]
    pipeline = Pipeline([
        # Use ColumnTransformer to combine the features
        ('union', ColumnTransformer(
            transformers=transformers,
            remainder='drop')),
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', RandomForestClassifier(random_state=5))
    ])

    return pipeline


###############################################################################
# Evaluation of different encoding methods
# -----------------------------------------
# We then loop over encoding methods, scoring the different pipeline predictions
# using a cross validation score:
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=3, random_state=12, shuffle=True)
all_scores = {}
for method in ['one-hot', 'similarity', 'minhash', 'gap']:
    pipeline = make_pipeline(method)
    # Now predict the census region of each participant
    scores = cross_val_score(pipeline, df, y, cv=cv)
    all_scores[method] = scores

    print('%s encoding' % method)
    print('Accuracy score:  mean: %.3f; std: %.3f\n'
          % (scores.mean(), scores.std()))

###############################################################################
# Plot the results
# ------------------
import seaborn
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 3))
ax = seaborn.boxplot(data=pd.DataFrame(all_scores), orient='h')
plt.ylabel('Encoding', size=20)
plt.xlabel('Prediction accuracy     ', size=20)
plt.yticks(size=20)
plt.tight_layout()

###############################################################################
# We can see that encoding the data using a SimilarityEncoder or MinhashEncoder
# instead of OneHotEncoder helps a lot in improving the cross validation score!
