r"""
Fitting scalable, non-linear models on data with dirty categories
=================================================================
Non-linear models form a very large model class. Among others, this class
includes:

* Neural Networks
* Tree-based methods such as Random Forests, and the very powerful Gradient
  Boosting Machines [#xgboost]_
* Kernel Methods.


Non-linear models can sometimes outperform linear ones, as they are able to
grasp more complex relationships between the input and the output of a machine
learning problem. However, using a non-linear model model often comes at a
cost:

* for neural networks, this cost is an extended model tuning time, in order to
  achieve good optimization and network architecture.
* gradient boosting machines do not tend to scale extremely well with
  increasing sample size, as all the data needs to be loaded into the main
  memory.
* for kernel methods, parameter fitting requires the inversion of a gram matrix
  of size :math:`n \times m` (:math:`n` being the number of samples), yiedling
  a quadratic dependency (with n) in the final compmutation time.


All is not lost though. For kernel methods, there exists approximation
algorithms, that drops the quadratic dependency with the sample size while
ensuring almost the same model capacity.

In this example, you will learn how to:
    1. build a ML pipeline that uses a kernel method.
    2. make this pipeline scalable.


.. note::
   This example assumes the reader to be familiar with similarity encoding and
   its use-cases.

   * For an introduction to dirty categories, see :ref:`this example<sphx_glr_auto_examples_01_dirty_categories.py>`.
   * To learn with dirty categories using the SimilarityEncoder, see :ref:`this example<sphx_glr_auto_examples_02_predict_employee_salaries.py>`.


.. |NYS| replace:: :class:`Nystroem <sklearn.kernel_approximation.Nystroem>`

.. |NYS_EXAMPLE|
    replace:: :ref:`scikit-learn documentation <nystroem_kernel_approx>`

.. |RBF|
    replace:: :class:`RBFSampler <sklearn.kernel_approximation.RBFSampler>`

.. |SE| replace:: :class:`SimilarityEncoder <dirty_cat.SimilarityEncoder>`

.. |SGDClassifier| replace::
    :class:`SGDClassifier <sklearn.linear_model.SGDClassifier>`

.. |APS| replace::
    :func:`average_precision_score <sklearn.metrics.average_precision_score>`

.. |OHE| replace::
    :class:`OneHotEncoder <sklearn.preprocessing.OneHotEncoder>`

.. |SGDClassifier_partialfit| replace::
    :func:`partial_fit <sklearn.linear_model.SGDClassifier.partial_fit>`

.. |Pipeline| replace::
    :class:`Pipeline <sklearn.pipeline.Pipeline>`

"""
###############################################################################
# Data Importing and preprocessing
# --------------------------------
#
# The data that the model will fit is the :code:`traffic violations` dataset.
import pandas as pd
from dirty_cat.datasets import fetch_traffic_violations

info = fetch_traffic_violations()
print(info['description'])

###############################################################################
# Problem Setting
# ---------------
# .. note::
#    We set the goal of our machine learning problem as follows: **predict the
#    violation type as a function of the description.**
#
# The :code:`Description` column, is composed of noisy textual observations. In
# the first part of this example, we will solely use this column as the input
# of our model. The output column is the :code:`Violation Type` column.  It
# consists of categorial values: therefore, our problem is a classification
# problem. You can have a glimpse of the values here:

df = pd.read_csv(info['path'], nrows=10).astype(str)
print(df[['Description', 'Violation Type']].head())
# This will be useful further down in the example.
columns_names = df.columns

###############################################################################
# Training a first simple pipeline
# --------------------------------
# Our input is categorical, thus needs to be encoded. As observations often
# consist in variations around a few concepts (for instance,
# :code:`'FAILURE OF VEH. ON HWY. TO DISPLAY LIGHTED LAMPS'` and
# :code:`'FAILURE TO DISPLAY TWO LIGHTED FRONT LAMPS WHEN REQUIRED'`
# are variations around not displaying lamps), we need an encoding able to
# capture similarities between observations.
from dirty_cat import SimilarityEncoder
similarity_encoder = SimilarityEncoder(similarity='ngram')

###############################################################################
# Once the input has be preprocessed, a choice has to be made regarding the
# kind of model used to fit the data. As said below, this example is about kernel methods, a good
# choice is to use a Support Vector Classifier, or :code:`SVC`:
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=42, gamma=1)

###############################################################################
# Those two estimators can then be stacked into a :code:`Pipeline`:
from sklearn.pipeline import Pipeline
steps = [('encoder', similarity_encoder), ('classifier', classifier)]
model = Pipeline(steps)

###############################################################################
# Model Evaluation
# ----------------
# Like in most machine learning setups, the data has to be splitted into 2 or
# more exclusive parts:
#
# * one for model training
# * one for model testing,
# * potentially, one for model selection/monitoring.
#
# For this reason, we create a simple wrapper around ``pandas.read_csv``, that
# extracts the :code:`X`, and :code:`y` from the dataset.
#
# .. note::
#    The :code:`y` labels are composed of 4 unique classes: ``Citation``,
#    ``Warning``, ``SERO``, and ``ESERO``. ``Citation`` and ``Warning``
#    represent around 99% of the data, in a fairly balanced manner. The last
#    two classes (``SERO`` and ``ESERO`` are very rare (1% and 0.1 % of the
#    data). Dealing with class imbalance is out of the scope of this example,
#    so the models will be trained on the first two classes only.
#
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# pre-fit the label encoder on the only-two classes it will see.
label_encoder = LabelEncoder()  # converts labels into integers
label_encoder.fit(['Citation', 'Warning'])

def get_X_y(**kwargs):
    """simple wrapper around pd.read_csv that extracts features and labels

    The y values are converted into integers to avoid doing this transformation
    repeatedly in the code.
    """
    global label_encoder
    df = pd.read_csv(info['path'], **kwargs)
    df = df.loc[df['Violation Type'].isin(['Citation', 'Warning'])]
    df = df.dropna()

    X, y = df[['Description']].values, df[['Violation Type']].values

    y_int = label_encoder.transform(np.squeeze(y))

    return X, y_int


###############################################################################
# To compute the |APS|, a binary version of the lables are needed, so a |OHE|
# is created to transform the integer labels into binary vectors.  This encoder
# fitted once with the only two values (0 and 1, since there are only two
# classes) it can see.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
one_hot_encoder = OneHotEncoder(categories="auto")  # integers to binary values
one_hot_encoder.fit([[0], [1]])

###############################################################################
# .. warning::
#    During the following training procedures of this example, we will assume
#    that the dataset was shuffled prior to its loading. As a reason, we can
#    take the first :math:`n` observations for the training set, the next
#    :math:`m` observations for the test set and so on. This may not be the
#    case for all datasets, so be careful before applying this code to your own
#    !
#
# Finally, the :code:`X` and :code:`y` are loaded.
#
# .. note::
#    We create an offset to separate the training and the test set. The reason
#    for this, is that the online procedures of this example will consume far
#    more rows, but we still would like to compare accuracies with the same the
#    same test set, and not change it each time. Therefore, we "reserve" the
#    first 100000 rows for the training phase. The rest is made available to
#    the test set.
train_set_size = 3000
test_set_size = 1000
cv_size = 1000
offset = 100000

X_train, y_train = get_X_y(skiprows=1, names=columns_names,
                           nrows=train_set_size)

X_test, y_test = get_X_y(skiprows=offset, names=columns_names,
                         nrows=test_set_size)
X_cv, y_cv = get_X_y(skiprows=offset+test_set_size, names=columns_names,
                     nrows=cv_size)

###############################################################################
# Evaluating time and sample complexity
# -------------------------------------
# Let's get an idea of model precision and performance depending on the number
# of the samples used in the train set.  First, a small helper is created to
# profile model fitting.
import time
def profile(func):
    def profiled_func(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        total_time = time.time() - t0
        return total_time, res
    return profiled_func

###############################################################################
# The |Pipeline| is then trained over different training set sizes. For this,
# :code:`X_train` and :code:`y_train` get sliced into subsets of increasing
# size, while :code:`X_test` and :code:`y_test` do not get changed when the
# sample size varies.
from sklearn.metrics import average_precision_score
# define the different train set sizes on which to evaluate the model
train_set_sizes = [train_set_size // 10, train_set_size // 3, train_set_size]

def evaluate_model(model, X_train, y_train, X_test, y_test, train_set_sizes):
    global one_hot_encoder
    train_times, test_scores = [], []

    for n in train_set_sizes:
        train_time, _ = profile(model.fit)(X_train[:n], y_train[:n])

        y_pred = model.predict(X_test)

        y_pred_onehot = one_hot_encoder.transform(
            y_pred.reshape(-1, 1)).toarray()
        y_test_onehot = one_hot_encoder.transform(
            y_test.reshape(-1, 1)).toarray()

        test_score = average_precision_score(y_test_onehot, y_pred_onehot)

        train_times.append(train_time)
        test_scores.append(test_score)
    return train_times, test_scores

train_times_svc, test_scores_svc = evaluate_model(model, X_train, y_train,
                                                  X_test, y_test,
                                                  train_set_sizes)

###############################################################################
# Plotting the training accuracy and the time:
import matplotlib.pyplot as plt
def plot_time_and_scores(
        train_set_sizes, train_times, test_scores, title=None):
    f, ax_time = plt.subplots()
    ax_time.set_ylabel('time')
    ax_time.set_xlabel('training set size')
    ax_time.plot(train_set_sizes, train_times, 'b-', label='time')
    ax_time.legend(loc='upper left')

    # display scores in the right-hand side y-axis
    ax_score = ax_time.twinx()
    ax_score.set_ylabel('score')
    ax_score.plot(train_set_sizes, test_scores, 'r-', label='score')
    ax_score.legend(loc='upper right')

    if title is not None:
        f.suptitle(title)
    return f

title = 'Test Accuracy and Fitting Time for a support vector classifier'
f = plot_time_and_scores(train_set_sizes, train_times_svc, test_scores_svc,
                         title=title)


###############################################################################
# Increasing training size cleary improves model accuracy. However, the
# training time and the input size increase quadratically with the training set
# size.  Moreover, kernel methods, process the entire :math:`n \times n` gram
# matrix at once, so the whole dataset needs to be loaded in the main memory at
# the same time. This is can be a strong limitation on the training set size:
# Using 30000 observations, the input is a :math:`30000 \times 30000` matrix.
# If composed of 32-bit floats, its total size is around :math:`30000^2 \times
# 32 = 2.8 \times 10^{10} \text{bits} = 4\text{GB}`, which is the caracteristic
# size of many personal laptops.


###############################################################################
# Kernel approximation methods
# ----------------------------
# Kernel approximation methods, such as |RBF| or |NYS| find a feature space in
# which the gram matrix of the samples is very similar to what the kernel
# matrix would be.  Thanks to this the problem can now be solved without a
# :math:`n \times n` matrix, but a :math:`n \times k`, :math:`k` being the
# dimension of the feature map matrix. More information can be found in the
# |NYS_EXAMPLE|

from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import LinearSVC
n_out_rbf = 100  # number of features after the RBFSampler transformation

###############################################################################
# Concretly, the difference with the previous model, is that now, a |RBF| or
# |NYS| object is inserted prior to the classifier, but after the |SE|:
linear_svc = LinearSVC(dual=False, random_state=42)
rbf_sampler = RBFSampler(n_components=n_out_rbf, random_state=42)
similarity_encoder = SimilarityEncoder(similarity='ngram')

steps = [('encoder', similarity_encoder),
         ('sampler', rbf_sampler),
         ('svc', linear_svc)]

pipeline_rbfs = Pipeline(steps)

###############################################################################
# The training procedure is then exactly the same:
train_times_rbfs, test_scores_rbfs = evaluate_model(
    pipeline_rbfs, X_train, y_train, X_test, y_test, train_set_sizes)
title = 'Test Accuracy and Fitting Time for SVC+RBFSampler'
f = plot_time_and_scores(train_set_sizes, train_times_rbfs, test_scores_rbfs,
                         title=title)


###############################################################################
# Two remarks:
#
# * We dropped the quadratic dependency of time with input size.
# * We did this at a minimal cost: the accuracy scores are very similar!
#

###############################################################################
# Online training
# ---------------
# .. note::
#    An online algorithm [#online_ref]_ is an algorithm that treats its input
#    piece by piece in a serial fashion. An famous example is the stochastic
#    gradient descent [#sgd_ref]_, where an estimation the objective function's
#    gradient is computed on a chunk of the data at each step.
#
# Kernel methods create a feature map by defining the scalar products of two
# elements in this feature map. But the feature map is not explicitally known,
# or even if known, can be of too big dimension to be computed! The magic of
# kernel methods is that when solving what is called the **dual problem** ,
# (the original optimization problem being the **primal problem**),only the
# gram matrix (matrix of the scalar product between every pair of the training
# observation) is needed. Fortunately, for a fair amount of interesting cost
# functions (including the ones we use here), the solution of the dual problem
# and of the primal problem is the same. So kernel methods can solve an
# optimization in a particular space, without even computing or knowing the
# features in this space!
#
# The price to pay though, is that the data cannot be processed by pieces,
# because the optimization of the dual problem is done by inverting the gram
# matrix of the inputs: all the data has to be processed at once.
#
# Kernel approximations relase this constraint by finding a feature map whose
# gram matrix (evaluated on the training set) approximate the gram matrix of
# the original kernel.  Now that the feature map is known, we can go back to
# solving the primal problem instead, and use the *online* optimization
# algorithm we want!  Let's take, for example, a |SGDClassifier|:
from sklearn.linear_model.stochastic_gradient import SGDClassifier
n_out_rbf = 100  # number of features RBFSampler feature map

# Filter annoying warning on max_iter and tol
import warnings
warnings.filterwarnings('ignore', module='sklearn.linear_model')


similarity_encoder = SimilarityEncoder(similarity='ngram')
rbf_sampler = RBFSampler(n_components=n_out_rbf, random_state=42)
sgd_classifier = SGDClassifier(max_iter=1000, tol=None, random_state=42)

steps = [('encoder', similarity_encoder),
         ('sampler', rbf_sampler),
         ('svc', sgd_classifier)]

pipeline_sgd = Pipeline(steps)

###############################################################################
# Again, the model evaluation procedure is exactly the same:
train_times_sgd, test_scores_sgd = evaluate_model(
    pipeline_sgd, X_train, y_train, X_test, y_test, train_set_sizes)
title = 'Test Accuracy and Fitting Time for SGDClassifier+RBFSampler'
f = plot_time_and_scores(train_set_sizes, train_times_sgd, test_scores_sgd,
                         title=title)

###############################################################################
# Reducing encoding dimension
# ---------------------------
# As the dataset on which the models are trained contains dirty categorical
# data, the cardindality of the categories may rise to a point where the
# intermediate feature matrix given by the |SE| becomes too big to hold into
# main memory.
#
# For this reason, it may be desirable to fit the similarity encoder
# using a fixed set of categories.
import numpy as np
n_out_encoder = 100
n_out_rbf = 100
categories = X_train[:n_out_encoder]

###############################################################################
# If specifying categories, they need to be sorted, per column, so there is a
# little bit of pre-processing to do:
def get_categories(X):
    """extract sorted observation, per column"""
    X = np.sort(X, axis=0)  # sort X per column
    return X.T.tolist()

categories = get_categories(categories)
similarity_encoder = SimilarityEncoder(similarity='ngram',
                                       categories=categories)
rbf_sampler = RBFSampler(n_components=n_out_rbf, random_state=42)
sgd_classifier = SGDClassifier()

steps = [('encoder', similarity_encoder),
         ('sampler', rbf_sampler),
         ('svc', sgd_classifier)]
pipeline_fixed_categories = Pipeline(steps)

train_times_fixed_cat, test_scores_fixed_cat = [], []

train_times_fixed_cat, test_scores_fixed_cat = evaluate_model(
    pipeline_fixed_categories, X_train, y_train, X_test, y_test,
    train_set_sizes)

title = ('test accuracy and fitting time for truncated '
         '\nSimilarityEncoder+SGDClassifier+RBFSampler')
f = plot_time_and_scores(train_set_sizes, train_times_fixed_cat,
                         test_scores_fixed_cat, title=title)

###############################################################################
# Online training for out-of-memory data
# --------------------------------------
# In the previous examples, the dataset was still entirely loaded in the main
# memory. For larger train sizes, it may not be possible. Fortunately,
# |SGDClassifier| objects implements the |SGDClassifier_partialfit| method. As
# a result, one can load one chunk of the full dataset, do a
# |SGDClassifier_partialfit| on it, discard the chunk out of memory, and so on
# until all the data passes through the dataset. For this though, there is a
# little bit more code to write.
#
# * first, pipeline objects do not implement |SGDClassifier_partialfit|, so it
#   is not possible to plug scikit-learn objects one after each other, for
#   online training the preprocessing and the model fitting part have to be
#   done separately.
# * second, in the previous cases, |RBF| and the |SE| were fitted only once,
#   right before the classifier fitting. Now, batch fitting is done, but the
#   representation of the samples cannot change for each batch. We have to
#   stick to one feature map, chosen at the beginning of the training phase.
#   More precisely, the categories of the |SE| will be chosen as a fixed subset
#   of the data. As for the fitting process of the RBFSampler, an good first
#   try can be the transformed features of the categories of the |SE|
#
# Once again, one of the conditions of the |SGDClassifier| convergence is using
# properly shuffled data. The data used in this example has been shuffled,
# prior the execution of this notebook. In order to get good results, you must
# make sure this holds in your local environment.


###############################################################################
# define helper to get the number of lines of the dataset
def file_length(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
n_total_samples = file_length(info['path'])

###############################################################################
# Here are defined the encoder's dimensions, and the datasets size.  In this
# case, we will monitor the evolution of the cross-validation accuracy over
# epochs, so we need a cross-validation set.
online_train_set_size = 10000
n_out_encoder = 100
rbf_sampler_out_dim = 100


###############################################################################
# The |SE| and the |RBF| are fitted once, at the beginning of the training
# procedure, on a subset of the dataset. As the dataset is shuffled, a
# reasonable choice for the |SE| categories is taking the first
# n_out_encoder rows.  The |RBF| can then be fitted on the similarity matrix
# given by the |SE|.

categories, _ = get_X_y(nrows=n_out_encoder, names=columns_names)
categories = get_categories(categories)

# Create and fit the similarity_encoder on its categories
similarity_encoder = SimilarityEncoder(
    similarity='ngram', categories=categories)
transformed_categories = similarity_encoder.fit_transform(
    np.reshape(categories, (-1, 1)))

# fit the rbf_sampler
rbf_sampler = RBFSampler(rbf_sampler_out_dim, random_state=42)
rbf_sampler.fit(transformed_categories)

###############################################################################
# create a |SGDClassifier|:
sgd_classifier = SGDClassifier(
    max_iter=1, tol=None, random_state=42,
    # learning_rate='constant',
    # eta0=0.01
)


###############################################################################
# The inputs and labels of the cv and test sets have to be pre-processed the
# same way the training set was processed:
y_true_cv_onehot = one_hot_encoder.transform(y_cv.reshape(-1, 1)).toarray()
y_true_test_onehot = one_hot_encoder.transform(y_test.reshape(-1, 1)).toarray()

X_test_sim_encoded = similarity_encoder.transform(X_test)
X_cv_sim_encoded = similarity_encoder.transform(X_cv)

X_test_kernel_approx = rbf_sampler.transform(X_test_sim_encoded)
X_cv_kernel_approx = rbf_sampler.transform(X_cv_sim_encoded)

###############################################################################
# We can now start the training. There are two intricated loop, one iterating
# over epochs (the number of passes of the |SGDClassifier| on the dataset, and
# on over chunks (the distincts pieces of the datasets).
chunksize = 1000
n_epochs = 5
for epoch_no in range(n_epochs):
    iter_csv = pd.read_csv(
        info['path'], nrows=online_train_set_size, chunksize=chunksize,
        skiprows=1, names=columns_names,
        usecols=['Description', 'Violation Type'])
    for chunk_no, chunk in enumerate(iter_csv):
        chunk = chunk.dropna(how='any')
        chunk = chunk.loc[~chunk['Violation Type'].isin(['SERO', 'ESERO'])]

        batch_X = chunk[['Description']].values
        batch_Y = chunk['Violation Type'].values

        # represent the labels as classes index
        batch_Y_integers = label_encoder.transform(batch_Y)
        batch_Y_onehot = one_hot_encoder.transform(
            batch_Y_integers.reshape(-1, 1)).toarray()

        # go through the preprocessing pipeline manually
        batch_X_sim_encoded = similarity_encoder.transform(batch_X)
        batch_X_kernel_approx = rbf_sampler.transform(batch_X_sim_encoded)

        # make one pass of stochastic gradient descent over the batch.
        sgd_classifier.partial_fit(
            batch_X_kernel_approx, batch_Y_integers,
            classes=list(range(len(label_encoder.classes_))))


        # print train/test accuracy metrics every 5 chunks
        if (chunk_no % 5) == 0:
            message = "chunk {:>4} epoch {:>3} ".format(chunk_no, epoch_no)
            for origin, X, y_true_onehot in zip(
                    ('train', 'cv'),
                    (batch_X_kernel_approx, X_cv_kernel_approx),
                    (batch_Y_onehot, y_true_cv_onehot)):

                # compute cross validation accuracty at the end of each epoch
                y_pred = sgd_classifier.predict(X)

                # preprocess correctly the labels and prediction to match
                # average_precision_score expectations
                y_pred_onehot = one_hot_encoder.transform(
                    y_pred.reshape(-1, 1)).toarray()

                score = average_precision_score(
                    y_true_onehot, y_pred_onehot, average=None)
                average_score = np.nanmean(score)

                message += "{} precision: {:.4f}  ".format(
                    origin, average_score)
            print(message)

# finally, compute the test score
y_test_pred = sgd_classifier.predict(X_test_kernel_approx)
y_test_pred_onehot = one_hot_encoder.transform(
    y_test_pred.reshape(-1, 1)).toarray()
test_score_online = average_precision_score(
    y_true_test_onehot, y_test_pred_onehot)
print('final test score for online method {:.4f}'.format(test_score_online))

###############################################################################
# Time and performance comparison
# -------------------------------
# We can finally compare the methods all together:
# We can now start the training. There are two intricated loop, one iterating
fig, ax = plt.subplots()
n_groups = 3
index = np.arange(n_groups)
bar_width = 0.1

opacity = 0.4

labels = ['svc', 'rbfs', 'sgd', 'fixed_cat']
test_scores = [test_scores_svc, test_scores_rbfs, test_scores_sgd,
               test_scores_fixed_cat]
colors = ['red', 'green', 'blue', 'orange']

i = -1
for label, scores, color in zip(labels, test_scores, colors):
    ax.bar(index + i*bar_width, scores, bar_width, alpha=opacity, color=color,
           label=label)
    i += 1

# plotting the online model accuracy
ax.axhline(test_score_online, linestyle='--', label='online model',
           color='black', alpha=opacity)

ax.set_xlabel('Train set size')
ax.set_ylabel('Test scores')
ax.set_title('Scores by model and train set size')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(list(map(str, train_set_sizes)))
ax.legend()

fig.tight_layout()
plt.show()

# over epochs (the number of passes of the |SGDClassifier| on the dataset, and
# on over chunks (the distincts pieces of the datasets).

###############################################################################
# .. rubric:: Footnotes
# .. [#xgboost] `Slides on gradient boosting by Tianqi Chen, the founder of XGBoost <https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf>`_
# .. [#online_ref] `Wikipedia article on online algorithms <https://en.wikipedia.org/wiki/Online_algorithm>`_
# .. [#sgd_ref] `Leon Bouttou's article on stochastic gradient descent <http://khalilghorbal.info/assets/spa/papers/ML_GradDescent.pdf>`_
