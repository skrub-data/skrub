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

* For neural networks, this cost is an extended model tuning time, in order to
  achieve good optimization and network architecture.
* Gradient Boosting Machines do not tend to scale extremely well with
  increasing sample size, as all the data needs to be loaded into the main
  memory.
* For kernel methods, parameter fitting requires the inversion of a gram matrix
  of size :math:`n \times n` (:math:`n` being the number of samples), yiedling
  a quadratic dependency (with n) in the final compmutation time.


All is not lost though. For kernel methods, there exist approximation
algorithms that drop the quadratic dependency with the sample size while
ensuring almost the same model capacity.

In this example, you will learn how to:
    1. build a ML pipeline that uses a kernel method.
    2. make this pipeline scalable, by using online algorithms and dimension
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
# The data that the model will fit is the :code:`traffic violations` dataset.
from dirty_cat.datasets import fetch_traffic_violations

info = fetch_traffic_violations()
print(info['description'])

###############################################################################
# .. topic:: Problem Setting
#
#    We set the goal of our machine learning problem as follows:
#
#    .. centered::
#       **predict the violation type as a function of the description.**
#
#
# The :code:`Description` column, is composed of noisy textual observations,
# while the :code:`Violation Type` column consists of categorial values:
# therefore, our problem is a classification problem. You can have a glimpse of
# the values here:
import pandas as pd

df = pd.read_csv(info['path'], nrows=10).astype(str)
print(df[['Description', 'Violation Type']].head())
# This will be useful further down in the example.
columns_names = df.columns

###############################################################################
# Estimators construction
# -----------------------
# Our input is categorical, thus needs to be encoded. As observations often
# consist in variations around a few concepts (for instance,
# :code:`'FAILURE OF VEH. ON HWY. TO DISPLAY LIGHTED LAMPS'` and
# :code:`'FAILURE TO DISPLAY TWO LIGHTED FRONT LAMPS WHEN REQUIRED'`
# are variations around not displaying lamps), we need an encoding able to
# capture similarities between observations.
from dirty_cat import SimilarityEncoder
similarity_encoder = SimilarityEncoder(similarity='ngram')

###############################################################################
# We can now choose a kernel method, for instance a |SVC|, to fit our data, and
# stack those two objects into a |Pipeline|.
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

classifier = SVC(kernel='rbf', random_state=42, gamma=1)
steps = [('encoder', similarity_encoder), ('classifier', classifier)]
model = Pipeline(steps)

###############################################################################
# Data Loading and Preprocessing
# ------------------------------
# Like in most machine learning setups, the data has to be splitted into 2 or
# more exclusive parts:
#
# * One for model training.
# * One for model testing.
# * Potentially, one for model selection/monitoring.
#
# For this reason, we create a simple wrapper around |pd_read_csv|, that
# extracts the :code:`X`, and :code:`y` from the dataset.
#
# .. topic:: Note about class imbalance:
#
#    The :code:`y` labels are composed of 4 unique classes: ``Citation``,
#    ``Warning``, ``SERO``, and ``ESERO``. ``Citation`` and ``Warning``
#    represent around 99% of the data, in a fairly balanced manner. The last
#    two classes (``SERO`` and ``ESERO``) are very rare (1% and 0.1 % of the
#    data). Dealing with class imbalance is out of the scope of this example,
#    so the models will be trained on the first two classes only.


def preprocess(df):
    global label_encoder
    df = df.loc[df['Violation Type'].isin(['Citation', 'Warning'])]
    df = df.dropna()

    X, y = df[['Description']].values, df[['Violation Type']].values

    y_int = label_encoder.transform(np.squeeze(y))

    return X, y_int


def get_X_y(**kwargs):
    """simple wrapper around pd.read_csv that extracts features and labels

    Some systematic preprocessing is also carried out to avoid doing this
    transformation repeatedly in the code.
    """
    df = pd.read_csv(info['path'], **kwargs)
    return preprocess(df)

###############################################################################
# Classifier objects in |sklearn| often require
# :code:`y` to be integer labels.  Additionaly, |APS| requires a binary version
# of the labels.  For these two purposes, we create:
#
# * a |LabelEncoder|, that we pre-fitted on the known :code:`y` classes
# * a |OneHotEncoder|, pre-fitted on the resulting integer labels.
#
# Their |transform| methods can the be called at appopriate times.
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder = LabelEncoder()
label_encoder.fit(['Citation', 'Warning'])

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
        t0 = time.perf_counter()
        res = func(*args, **kwargs)
        total_time = time.perf_counter() - t0
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

        y_pred_onehot = one_hot_encoder.transform(y_pred.reshape(-1, 1))
        y_test_onehot = one_hot_encoder.transform(y_test.reshape(-1, 1))

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
# size.  Indeed, kernel methods need to process an entire :math:`n \times n`
# matrix at once. In order for this matrix to be loaded into memory, :math:`n`
# has to remain low: Using 30000 observations, the input is a :math:`30000
# \times 30000` matrix.  If composed of 32-bit floats, its total size is around
# :math:`30000^2 \times 32 = 2.8 \times 10^{10} \text{bits} = 4\text{GB}`,
# which is the caracteristic size of many personal laptops.

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
#    Kernel methods address non-linear problems by creating a new feature space
#    in which (hopefully) the original problem becomes linear. However, this
#    feature space is not defined explicitly.  Is it defined through the inner
#    product of two elements of this space :math:`K(x_1, x_2)`.
#
#    Different formulas for the inner product can result in arbitrary complex
#    feature space. It can even be of infinite dimension! But the magic of
#    kernel methods is that those new features are never **actually computed**.
#    Indeed, the kernel tricks consists in solving what is called the **dual
#    problem** [#dual_ref]_, (the original optimization problem being the
#    **primal problem**). To do so, only the gram matrix (matrix of the inner
#    product between every pair of the training observation) :math:`K_{ij}` is
#    needed.  In the original feature space, :math:`K_{ij} = x_i^Tx_j`, but in
#    the new feature space, :math:`K_{ij}=K(x_i, x_j)`.
#
#    Fortunately, for a fair amount of interesting cost functions (including
#    the ones we use here), the solution of the dual problem and of the primal
#    problem is the same. So in summary, kernel methods can solve an
#    optimization in a particular space, without even computing or knowing the
#    features in this space!


###############################################################################
# Kernel approximation methods
# ----------------------------
# From what was said below, two criterions are limiting a kernel algorithm to
# scale:
#
# * It processes an matrix, whose size increases quadratically with the number
#   of samples.
# * During dual optimization, this matrix is **inverted**, meaning it has to be
#   loaded into main memory.
#
# Kernel approximation methods such as |RBF| or |NYS| [#nys_ref]_,  take a
# specific kernel :math:`K` as an input, and, try to find the best
# d-dimensional feature space :math:`\Phi_d` in which the inner product of two
# elements matches K as much as possible:
#
# .. math::
#    \Phi_d(x_i)^T\Phi_d(x_j) \simeq K(x_i, x_j)
#
# Such algorithms address the two limitations at once:
#
# * Thanks to this, the problem can now be solved without a :math:`n \times n`
#   matrix, but a :math:`n \times d` one.
# * The only power of solving the dual problem instead of the primal one
#   was that the first one did not require the knowledge of :math:`\Phi`. Now
#   that :math:`\Phi` is known, solving the dual problem in not useful anymore,
#   and we can get back to the primal. This problem can now be treated as any
#   other machine learning problem, using for instance online optimization
#   algoriths, such as stochastic gradient descent.
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
n_out_encoder = 100
n_out_rbf = 100
n_samples_encoder = 10000

X_encoder, _ = get_X_y(nrows=n_samples_encoder, names=columns_names)

similarity_encoder = SimilarityEncoder(
    similarity='ngram', categories='most_frequent', n_prototypes=n_out_encoder)
transformed_categories = similarity_encoder.fit_transform(X_encoder)

# Fit the rbf_sampler with the similarity matrix.
rbf_sampler = RBFSampler(n_out_rbf, random_state=42)
rbf_sampler.fit(transformed_categories)


# The inputs and labels of the cv and test sets have to be pre-processed the
# same way the training set was processed:
def encode(X, y_int):
    global one_hot_encoder, similarity_encoder, rbf_sampler
    X_sim_encoded = similarity_encoder.transform(X)
    X_kernel_approx = rbf_sampler.transform(X_sim_encoded)
    y_onehot = one_hot_encoder.transform(y_int.reshape(-1, 1))
    return X_kernel_approx, y_onehot


X_cv_kernel_approx, y_true_cv_onehot = encode(X_cv, y_cv)
X_test_kernel_approx, y_true_test_onehot = encode(X_test, y_test)

###############################################################################
# Online training for out-of-memory data
# --------------------------------------
# We now have all the theoretical elements to create an non-linear, online
# kernel method.
import warnings
from sklearn.linear_model.stochastic_gradient import SGDClassifier

online_train_set_size = 50000
# Filter warning on max_iter and tol
warnings.filterwarnings('ignore', module='sklearn.linear_model')
sgd_classifier = SGDClassifier(max_iter=1, tol=None, random_state=42)

###############################################################################
# We can now start the training. There are two intricated loops, one iterating
# over epochs (the number of passes of the |SGDClassifier| on the dataset, and
# one over batches (the distincts pieces of the datasets).
batchsize = 1000
n_epochs = 5
for epoch_no in range(n_epochs):
    iter_csv = pd.read_csv(
        info['path'], nrows=online_train_set_size, chunksize=batchsize,
        skiprows=1, names=columns_names,
        usecols=['Description', 'Violation Type'])
    for batch_no, batch in enumerate(iter_csv):
        X_batch, y_batch = preprocess(batch)
        X_batch_kernel_approx, y_batch_onehot = encode(X_batch, y_batch)

        # make one pass of stochastic gradient descent over the batch.
        sgd_classifier.partial_fit(
            X_batch_kernel_approx, y_batch, classes=[0, 1])

        # print train/test accuracy metrics every 5 batch
        if (batch_no % 5) == 0:
            message = "batch {:>4} epoch {:>3} ".format(batch_no, epoch_no)
            for origin, X, y_true_onehot in zip(
                    ('train', 'cv'),
                    (X_batch_kernel_approx, X_cv_kernel_approx),
                    (y_batch_onehot, y_true_cv_onehot)):

                # compute cross validation accuracy at the end of each epoch
                y_pred = sgd_classifier.predict(X)

                # preprocess correctly the labels and prediction to match
                # average_precision_score expectations
                y_pred_onehot = one_hot_encoder.transform(
                    y_pred.reshape(-1, 1))

                score = average_precision_score(y_true_onehot, y_pred_onehot)
                message += "{} precision: {:.4f}  ".format(origin, score)

            print(message)

# finally, compute the test score
y_test_pred = sgd_classifier.predict(X_test_kernel_approx)
y_test_pred_onehot = one_hot_encoder.transform(y_test_pred.reshape(-1, 1))
test_score_online = average_precision_score(
    y_true_test_onehot, y_test_pred_onehot)
print('final test score for online method {:.4f}'.format(test_score_online))

###############################################################################
# .. rubric:: Footnotes
# .. [#xgboost] `Slides on gradient boosting by Tianqi Chen, the founder of XGBoost <https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf>`_
# .. [#online_ref] `Wikipedia article on online algorithms <https://en.wikipedia.org/wiki/Online_algorithm>`_
# .. [#sgd_ref] `Leon Bouttou's article on stochastic gradient descent <http://khalilghorbal.info/assets/spa/papers/ML_GradDescent.pdf>`_
# .. [#nys_ref] |NYS_EXAMPLE|
# .. [#dual_ref] `Introduction to duality <https://pdfs.semanticscholar.org/0373/e7289a1978108d6455218160a529c85842c0.pdf>`_
