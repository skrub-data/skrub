.. _end_to_end_pipeline:

============================
End-to-end predictive models
============================

.. currentmodule:: skrub

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
