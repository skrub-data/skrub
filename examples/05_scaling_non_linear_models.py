r"""
Fitting scalable, non-linear models on data with dirty categories
=================================================================

A very classic dilemna when training a machine learning model consists in
choosing between using a linear model or a non-linear one.

Linear models are very well studied and understood. They are generally fast
and easy to optimize, and generate interpretable results. However, for some
problems with a complex relationship between the input and the output, linear
models reach their expressivity limit: whatever the number of samples the
training set may have, past some point, it's precision won't get any better.

Non-linear models, however, tend to *scale* better with sample size: they are
able to digest the information in the additional samples to get a better
estimate of the link between the input and the output.

Non-linear models form a very large model class. Among others, this class
includes:

* Neural Networks
* Tree-based methods such as Random Forests, and the very powerful Gradient
  Boosting Machines [#xgboost]_
* Kernel Methods.

However, reaching the phase where the non-linear model outpeforms the linear
one can be complicated. Indeed, a more complex models means often a longer
fitting/tuning process:

* Neural networks often necessitate extended model tuning time, in order to
  achieve good optimization and network architecture.
* Gradient Boosting Machines do not tend to scale extremely well with
  increasing sample size, as all the data needs to be loaded into the main
  memory.
* For kernel methods, parameter fitting requires the inversion of a gram matrix
  of size :math:`n \times n` (:math:`n` being the number of samples), yiedling
  a quadratic dependency (with n) in the final compmutation time.


In order to make the best out of a non-linear model, one has to **make it
scalable**. For kernel methods, there exist approximation algorithms that
drop the quadratic dependency with the sample size while ensuring almost the
same model capacity.

In this example, you will learn how to:
    1. Build a ML pipeline that uses a kernel method.
    2. Make this pipeline scalable, by using online algorithms and dimension
       reduction methods.


.. note::
   This example assumes the reader to be familiar with similarity encoding and
   its use-cases.

   * For an introduction to dirty categories, see :ref:`this example<sphx_glr_auto_examples_01_dirty_categories.py>`.
   * To learn with dirty categories using the SimilarityEncoder, see :ref:`this example<sphx_glr_auto_examples_02_predict_employee_salaries.py>`.


.. |NYS| replace:: :class:`Nystroem <sklearn.kernel_approximation.Nystroem>`

.. |NYS_EXAMPLE|
    replace:: :ref:`scikit-learn documentation <nystroem_kernel_approx>`

.. |RBF|
    replace:: :class:`~sklearn.kernel_approximation.RBFSampler`

.. |SVC|
    replace:: :class:`SupportVectorClassifier <sklearn.svm.SVC>`

.. |SE| replace:: :class:`~dirty_cat.SimilarityEncoder`

.. |SGDClassifier| replace::
    :class:`~sklearn.linear_model.SGDClassifier`

.. |APS| replace::
    :func:`~sklearn.metrics.average_precision_score`

.. |OneHotEncoder| replace::
    :class:`~sklearn.preprocessing.OneHotEncoder`

.. |ColumnTransformer| replace::
    :class:`~sklearn.compose.ColumnTransformer`

.. |LabelEncoder| replace::
    :class:`~sklearn.preprocessing.LabelEncoder`

.. |SGDClassifier_partialfit| replace::
    :meth:`~sklearn.linear_model.SGDClassifier.partial_fit`

.. |Pipeline| replace::
    :class:`~sklearn.pipeline.Pipeline`

.. |pd_read_csv| replace::
    :func:`pandas.read_csv`

.. |sklearn| replace::
    :std:doc:`scikit-learn <sklearn:index>`

.. |transform| replace::
    :std:term:`transform <sklearn:transform>`

.. |fit| replace::
    :std:term:`fit <sklearn:fit>`

"""
###############################################################################
# Training a first simple pipeline
# --------------------------------
# The data that the model will fit is the :code:`drug_directory` dataset.
from dirty_cat.datasets import fetch_drug_directory

info = fetch_drug_directory()
print(info['description'])

###############################################################################
# .. topic:: Problem Setting
#
#    We set the goal of our machine learning problem as follows:
#
#    .. centered::
#       **predict the type of a drug given its composition.**
#
#
# The :code:`NONPROPRIETARYNAME` column, is composed of text observations with
# describing each drug's composition. The :code:`PRODUCTTYPENAME` column
# consists of categorial values: therefore, our problem is a classification
# problem. You can have a glimpse of the values here:
import pandas as pd

df = pd.read_csv(info['path'], nrows=10, sep='\t').astype(str)
print(df[['NONPROPRIETARYNAME', 'PRODUCTTYPENAME']].head())
# This will be useful further down in the example.
columns_names = df.columns

###############################################################################
# Estimators construction
# -----------------------
# Our input is categorical, thus needs to be encoded. As observations often
# consist in variations around a few concepts (for instance,
# :code:`'Amlodipine Besylate'` and
# :code:`'Amlodipine besylate and atorvastatin calcium'`
# have one ingredient in common), we need an encoding able to
# capture similarities between observations.

from dirty_cat import SimilarityEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
similarity_encoder = SimilarityEncoder(similarity='ngram')

###############################################################################
# Two other columns are used to predict the output: ``DOSAGEFORMNAME`` and
# ``ROUTENAME``. They are both categorical and can be encoded with a
# |OneHotEncoder|. We use a |ColumnTransformer| to stack the |OneHotEncoder|
# and the |SE|.  We can now choose a kernel method, for instance a |SVC|, to
# fit the encoded inputs.
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

column_transformer = make_column_transformer(
    (similarity_encoder, ['NONPROPRIETARYNAME']),
    (OneHotEncoder(handle_unknown='ignore'), ['DOSAGEFORMNAME', 'ROUTENAME']),
    sparse_threshold=1)

# The classifier and the ColumnTransformer are stacked into a Pipeline object
classifier = SVC(kernel='rbf', random_state=42, gamma=1)
steps = [('transformer', column_transformer), ('classifier', classifier)]
model = Pipeline(steps)

###############################################################################
# Data Loading and Preprocessing
# ------------------------------
# Like in most machine learning setups, the data has to be splitted into 2
# exclusive parts:
#
# * One for model training.
# * One for model testing.
#
# For this reason, we create a simple wrapper around |pd_read_csv|, that
# extracts the :code:`X`, and :code:`y` from the dataset.
#
# .. topic:: Note about class imbalance:
#
#    The :code:`y` labels are composed of 7 unique classes. However, ``HUMAN
#    OTC DRUG`` and ``HUMAN PRESCRIPTION DRUG`` represent around 97% of the
#    data, in a fairly balanced manner. The last 5 classes are much rarer.
#    Dealing with class imbalance is out of the scope of this example, so the
#    models will be trained on the first two classes only.


def preprocess(df, label_encoder):
    df = df.loc[df['PRODUCTTYPENAME'].isin(
        ['HUMAN OTC DRUG', 'HUMAN PRESCRIPTION DRUG'])]

    df = df[['NONPROPRIETARYNAME', 'DOSAGEFORMNAME', 'ROUTENAME', 'PRODUCTTYPENAME']]
    df = df.dropna()

    X = df[['NONPROPRIETARYNAME', 'DOSAGEFORMNAME', 'ROUTENAME']]
    y = df[['PRODUCTTYPENAME']].values

    y_int = label_encoder.transform(np.squeeze(y))

    return X, y_int


def get_X_y(**kwargs):
    """simple wrapper around pd.read_csv that extracts features and labels

    Some systematic preprocessing is also carried out to avoid doing this
    transformation repeatedly in the code.
    """
    global label_encoder
    df = pd.read_csv(info['path'], sep='\t', **kwargs)
    return preprocess(df, label_encoder)

###############################################################################
# Classifier objects in |sklearn| often require :code:`y` to be integer labels.
# Additionally, |APS| requires a binary version of the labels.  For these two
# purposes, we create:
#
# * a |LabelEncoder|, that we pre-fitted on the known :code:`y` classes
# * a |OneHotEncoder|, pre-fitted on the resulting integer labels.
#
# Their |transform| methods can the be called at appopriate times.
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder = LabelEncoder()
label_encoder.fit(['HUMAN OTC DRUG', 'HUMAN PRESCRIPTION DRUG'])

one_hot_encoder = OneHotEncoder(categories="auto", sparse=False)
one_hot_encoder.fit([[0], [1]])

###############################################################################
# .. warning::
#
#    During the following training procedures of this example, we will assume
#    that the dataset was shuffled prior to its loading. As a reason, we can
#    take the first :math:`n` observations for the training set, the next
#    :math:`m` observations for the test set and so on. This may not be the
#    case for all datasets, so be careful before applying this code to your own
#    !
#
#
# Finally, the :code:`X` and :code:`y` are loaded.
#
#
# .. topic:: Note: offsetting the test set
#
#    We create an offset to separate the training and the test set. The reason
#    for this, is that the online procedures of this example will consume far
#    more rows, but we still would like to compare accuracies with the same the
#    same test set, and not change it each time. Therefore, we "reserve" the
#    first 100000 rows for the training phase. The rest is made available to
#    the test set.
train_set_size = 5000
test_set_size = 10000
offset = 100000

X_train, y_train = get_X_y(skiprows=1, names=columns_names,
                           nrows=train_set_size)

X_test, y_test = get_X_y(skiprows=offset, names=columns_names,
                         nrows=test_set_size)

###############################################################################
# Evaluating time and sample complexity
# -------------------------------------
# Let's get an idea of model precision and performance depending on the number
# of the samples used in the train set.
# The |Pipeline| is trained over different training set sizes. For this,
# :code:`X_train` and :code:`y_train` get sliced into subsets of increasing
# size, while :code:`X_test` and :code:`y_test` do not change when the
# sample size varies.
import time
from sklearn.metrics import average_precision_score

# define the different train set sizes on which to evaluate the model
train_set_sizes = [train_set_size // 10, train_set_size // 3, train_set_size]

train_times_svc, test_scores_svc = [], []

for n in train_set_sizes:

    t0 = time.perf_counter()
    model.fit(X_train[:n], y_train[:n])
    train_time = time.perf_counter() - t0

    y_pred = model.predict(X_test)

    y_pred_onehot = one_hot_encoder.transform(y_pred.reshape(-1, 1))
    y_test_onehot = one_hot_encoder.transform(y_test.reshape(-1, 1))

    test_score = average_precision_score(y_test_onehot, y_pred_onehot)

    train_times_svc.append(train_time)
    test_scores_svc.append(test_score)

    msg = ("using {:>5} samples: model fitting took {:.1f}s, test accuracy of "
           "{:.3f}")
    print(msg.format(n, train_time, test_score))


###############################################################################
# Increasing training size cleary improves model accuracy. However, the
# training time and the input size increase quadratically with the training set
# size.  Indeed, kernel methods need to process an entire :math:`n \times n`
# matrix at once. In order for this matrix to be loaded into memory, :math:`n`
# has to remain low: Using 30000 observations, the input is a :math:`30000
# \times 30000` matrix.  If composed of 32-bit floats, its total size is around
# :math:`30000^2 \times 32 = 2.8 \times 10^{10} \text{bits} = 4\text{GB}`

###############################################################################
# Reducing input dimension using kernel approximation methods.
# ------------------------------------------------------------
#
# The main scalability issues with kernels methods is the processing of a large
# square matrix. To understand where this matrix comes from, we need to delve a
# little bit deeper these methods internals.
#
# .. topic:: Kernel methods
#
#    Kernel methods address non-linear problems by leveraging similarities
#    between each pair of inputs. Using a similarity matrix to solve a machine
#    learning problem allows to catch complex, non-linear relationships within
#    the data.  But it requires inverting this matrix, which can be a
#    computational burden when the sample sizes increases.


###############################################################################
# Kernel approximation methods
# ----------------------------
# From what was said below, two criterions are limiting a kernel algorithm to
# scale:
#
# * It processes a matrix, whose size increases quadratically with the number
#   of samples.
# * During fitting time, this matrix is **inverted**, meaning it has to be
#   loaded into main memory.
#
# Kernel approximation methods such as |RBF| or |NYS| [#nys_ref]_ try to
# approximate this similarity matrix, without actually creating it. By allowing
# the program to not compute the perfect similarity matrix, the problem
# complexity becomes linear! Plus, the samples also do not need to be processed
# at once into main memory. We are not bound to use a |SVC| anymore, and can
# instead use an online optimization that will process the input by batch.
#
# .. topic:: Online algorithms
#
#    An online algorithm [#online_ref]_ is an algorithm that treats its input
#    piece by piece in a serial fashion. An famous example is the stochastic
#    gradient descent [#sgd_ref]_, where an estimation the objective function's
#    gradient is computed on a batch of the data at each step.
#
#

###############################################################################
# Reducing the transformers dimensionality
# ----------------------------------------
# There is one last scalability issue in our pipeline: the |SE| and the |RBF|
# both implement the |fit| method. How to adapt those method to an online
# setting, where the data is never loaded as a whole?
#
# A simple solution is to partially fit the |SE| and the |RBF| on a subset of
# the data, prior to the online fitting step.

from sklearn.kernel_approximation import RBFSampler
n_out_encoder = 1000
n_out_rbf = 5000
n_samples_encoder = 10000

X_encoder, _ = get_X_y(nrows=n_samples_encoder, names=columns_names)

similarity_encoder = SimilarityEncoder(
    similarity='ngram', categories='most_frequent', n_prototypes=n_out_encoder,
    random_state=42, ngram_range=(2, 4))

# Fit the rbf_sampler with the similarity matrix.
column_transformer = make_column_transformer(
    (similarity_encoder, ['NONPROPRIETARYNAME']),
    (OneHotEncoder(handle_unknown='ignore'), ['DOSAGEFORMNAME', 'ROUTENAME']),
    sparse_threshold=1)

transformed_categories = column_transformer.fit_transform(X_encoder)

# gamma is a parameter of the rbf function, that sets how fast the similarity
# between two points should decrease as the distance between them rises. It
# is data-specific, and needs to be chosen carefully, for example using
# cross-validation.
rbf_sampler = RBFSampler(
    gamma=0.5, n_components=n_out_rbf, random_state=42)
rbf_sampler.fit(transformed_categories)


def encode(X, y_int, one_hot_encoder, column_transformer, rbf_sampler):
    X_sim_encoded = column_transformer.transform(X)

    X_highdim = rbf_sampler.transform(X_sim_encoded.toarray())

    y_onehot = one_hot_encoder.transform(y_int.reshape(-1, 1))

    return X_highdim, y_onehot


# The inputs and labels of the val and test sets have to be pre-processed the
# same way the training set was processed:
X_test_kernel_approx, y_true_test_onehot = encode(
    X_test, y_test, one_hot_encoder, column_transformer, rbf_sampler)


###############################################################################
# Online training for out-of-memory data
# --------------------------------------
# We now have all the theoretical elements to create an non-linear, online
# kernel method.
import warnings
from sklearn.linear_model.stochastic_gradient import SGDClassifier

online_train_set_size = 100000
# Filter warning on max_iter and tol
warnings.filterwarnings('ignore', module='sklearn.linear_model')
sgd_classifier = SGDClassifier(
    max_iter=1, tol=None, random_state=42, average=10)


###############################################################################
# We can now start the training, by looping over batches one by one. Note that
# only one pass over the whole dataset is done. It may be worth doing several
# passes, but for very large sample sizes, the increase in test accuracy is
# likely to be marginal.
batchsize = 1000
test_scores_rbf = []
train_times_rbf = []
online_train_set_sizes = []
t0 = time.perf_counter()

iter_csv = pd.read_csv(
    info['path'], nrows=online_train_set_size, chunksize=batchsize,
    skiprows=1, names=columns_names, sep='\t')

for batch_no, batch in enumerate(iter_csv):
    X_batch, y_batch = preprocess(batch, label_encoder)
    X_batch_kernel_approx, y_batch_onehot = encode(
        X_batch, y_batch, one_hot_encoder, column_transformer, rbf_sampler)

    # make one pass of stochastic gradient descent over the batch.
    sgd_classifier.partial_fit(
        X_batch_kernel_approx, y_batch, classes=[0, 1])

    # print train/test accuracy metrics every 5 batch
    if (batch_no % 5) == 0:
        message = "batch {:>4} ".format(batch_no)
        for origin, X, y_true_onehot in zip(
                ('train', 'val'),
                (X_batch_kernel_approx, X_test_kernel_approx),
                (y_batch_onehot, y_true_test_onehot)):

            y_pred = sgd_classifier.predict(X)

            # preprocess correctly the labels and prediction to match
            # average_precision_score expectations
            y_pred_onehot = one_hot_encoder.transform(
                y_pred.reshape(-1, 1))

            score = average_precision_score(y_true_onehot, y_pred_onehot)
            message += "{} precision: {:.4f}  ".format(origin, score)
            if origin == 'val':
                test_scores_rbf.append(score)
                train_times_rbf.append(time.perf_counter() - t0)
                online_train_set_sizes.append((batch_no + 1)*batchsize)

        print(message)

###############################################################################
# So far, we fitted two kinds of models: a exact kernel algorithm, and an
# approximate, online one. Lets compare both the accuracies and the number of
# visited samples for each model as we increase our time budget:
import matplotlib.pyplot as plt

f, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
ax_score, ax_capacity = axs

ax_score.set_ylabel('score')
ax_capacity.set_ylabel('training set size')
ax_capacity.set_xlabel('time')

ax_score.plot(train_times_svc, test_scores_svc, 'b-', label='exact')
ax_score.plot(train_times_rbf, test_scores_rbf, 'r-', label='online')

ax_capacity.plot(train_times_svc, train_set_sizes, 'b-', label='exact')
ax_capacity.plot(train_times_rbf, online_train_set_sizes, 'r-', label='online')
ax_capacity.set_yscale('log')

ax_score.legend(
    bbox_to_anchor=(0., 1.02, 1., .102),
    loc=3, ncol=2, mode='expand',
    borderaxespad=0.)

# compare the two methods in their common time range
ax_score.set_xlim(0, min(train_times_svc[-1], train_times_rbf[-1]))

title = """Test set accuracy and number of samples visited
samples seen by the SGDClassifier"""
f.suptitle(title)


###############################################################################
# This plot shows us that for the time budget, the online model will eventually
# process more samples, be faster and reach a far higher test accuracy that the
# non-online, exact kernel method.
#
# Our online model also outperforms online **linear** models (for instance,
# |SE| + |SGDClassifier|). We did not fit two online models here for simplicity
# purposes, but to train an online linear model, simply comment out the line in
# encode :code:`X_highdim = rbf_sampler.transform(X_sim_encoded.toarray())` and
# change it for example with :code:`X_highdim = X_sim_encoded`.
#
# In particular, this hierarchy between the linear model and the non-linear
# one shows that there were some significant non-linear relashionships
# between the input and the output. By scaling a kernel method, we succesfully
# took this non-linearity into account in our model, which was a far from
# trivial task at the beginning of this example!

###############################################################################
# .. rubric:: Footnotes
# .. [#xgboost] `Slides on gradient boosting by Tianqi Chen, the founder of XGBoost <https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf>`_
# .. [#online_ref] `Wikipedia article on online algorithms <https://en.wikipedia.org/wiki/Online_algorithm>`_
# .. [#sgd_ref] `Leon Bouttou's article on stochastic gradient descent <http://khalilghorbal.info/assets/spa/papers/ML_GradDescent.pdf>`_
# .. [#nys_ref] |NYS_EXAMPLE|
# .. [#dual_ref] `Introduction to duality <https://pdfs.semanticscholar.org/0373/e7289a1978108d6455218160a529c85842c0.pdf>`_
