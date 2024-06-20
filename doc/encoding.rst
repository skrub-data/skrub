.. _encoding:

====================================
Encoding: creating feature matrices
====================================

.. currentmodule:: skrub

Encoding or vectorizing creates numerical features from the data,
converting dataframes, strings, dates... Different encoders are suited
for different types of data.

.. _table_vectorizer:

Turning a dataframe into a numerical feature matrix
---------------------------------------------------

A dataframe can comprise columns of all kind of types. A good numerical
representation of these columns help analytics and statistical learning.

The :class:`TableVectorizer` gives a turn-key solution by applying
different data-specific encoder to the different columns. It makes reasonable
heuristic choices that are not necessarily optimal since it is not aware of the learner
used for the machine learning task). However, it already provides a typically very good
baseline.

The function :func:`tabular_learner` goes the extra mile by creating a machine-learning
model that works well on tabular data. This model combines a :class:`TableVectorizer`
with a provided scikit-learn estimator. Depending whether or not the final estimator
natively support missing values, a missing value imputer step is added before the
final estimator. The parameters of the :class:`TableVectorizer` are chosen based on the
type of the final estimator.

.. list-table:: Parameter values choice of :class:`TableVectorizer` when using the :func:`tabular_learner` function
   :header-rows: 1

   * -
     - ``RandomForest`` models
     - ``HistGradientBoosting`` models
     - Linear models and others
   * - Low-cardinality encoder
     - :class:`~sklearn.preprocessing.OrdinalEncoder`
     - Native support :sup:`(1)`
     - :class:`~sklearn.preprocessing.OneHotEncoder`
   * - High-cardinality encoder
     - :class:`MinHashEncoder`
     - :class:`MinHashEncoder`
     - :class:`GapEncoder`
   * - Numerical preprocessor
     - No processing
     - No processing
     - :class:`~sklearn.preprocessing.StandardScaler`
   * - Date preprocessor
     - :class:`DatetimeEncoder`
     - :class:`DatetimeEncoder`
     - :class:`DatetimeEncoder`
   * - Missing value strategy
     - Native support :sup:`(2)`
     - Native support
     - :class:`~sklearn.impute.SimpleImputer`

.. note::
  :sup:`(1)` if scikit-learn installed is lower than 1.4, then
  :class:`~sklearn.preprocessing.OrdinalEncoder` is used since native support
  for categorical features is not available.

  :sup:`(2)` if scikit-learn installed is lower than 1.4, then
  :class:`~sklearn.impute.SimpleImputer` is used since native support
  for missing values is not available.

With tree-based models, the :obj:`MinHashEncoder` is used for high-cardinality
categorical features. It does not provide interpretable features as the default
:obj:`GapEncoder` but it is much faster. For low-cardinality, these models relies on
either the native support of the model or the
:obj:`~sklearn.preprocessing.OrdinalEncoder`.

With linear models or unknown models, the default values of the different
parameters are used. Therefore, the :obj:`GapEncoder` is used for
high-cardinality categorical features and the
:obj:`~sklearn.preprocessing.OneHotEncoder` for low-cardinality ones. If the
final estimator does not support missing values, a
:obj:`~sklearn.impute.SimpleImputer` is added before the final estimator.
Finally, a :obj:`~sklearn.preprocessing.StandardScaler` is added to the
pipeline. Those choices may not be optimal in all cases but they are
methodologically safe.

.. _dirty_categories:

Encoding open-ended entries and dirty categories
------------------------------------------------

String columns can be seen categories for statistical analysis, but
standard tools to represent categories fail if these strings are not
normalized into a small number of well-identified form, if they have
typos, or if there are too many categories.

Skrub provides encoders that represent well open-ended strings or dirty
categories, eg to replace :class:`~sklearn.preprocessing.OneHotEncoder`:

* :class:`GapEncoder`: infers latent categories and represent the data on
  these. Very interpretable, sometimes slow

* :class:`MinHashEncoder`: a very scalable encoding of strings capturing
  their similarities. Particularly useful on large databases and well
  suited for learners such as trees (boosted trees or random forests)

* :class:`SimilarityEncoder`: a simple encoder that works by representing
  strings similarities with all the different categories in the data.
  Useful when there are a small number of categories, but we still want
  to capture the links between them (eg: "west", "north", "north-west")

.. topic:: References::

    For a detailed description of the problem of encoding dirty
    categorical data, see `Similarity encoding for learning with dirty
    categorical variables <https://hal.inria.fr/hal-01806175>`_ [1]_ and
    `Encoding high-cardinality string categorical variables
    <https://hal.inria.fr/hal-02171256v4>`_ [2]_.

.. [1] Patricio Cerda, Gaël Varoquaux.
       Encoding high-cardinality string categorical variables. 2020.
       IEEE Transactions on Knowledge & Data Engineering.
.. [2] Patricio Cerda, Gaël Varoquaux, Balázs Kégl.
       Similarity encoding for learning with dirty categorical variables. 2018.
       Machine Learning journal, Springer.

Encoding dates
---------------

The :class:`DatetimeEncoder` encodes date and time: it represent them as
time in seconds since a fixed date, but also added features useful to
capture regularities: week of the day, month of the year...
