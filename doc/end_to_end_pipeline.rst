.. _end_to_end_pipeline:

============================
End-to-end predictive models
============================

.. currentmodule:: skrub

.. _tabular_learner:

Create baseline predictive models on heterogeneous dataset
----------------------------------------------------------

Crafting a machine-learning pipeline is rather a daunting task. Choosing the ending
learner of such pipeline is usually the easiest part. However, it imposes constraints
regarding the preprocessing steps that are are required ahead of the learner.
Programmatically defining these steps is the part that requires the most expertise and
that is cumbersome to write.

The function :func:`tabular_learner` provides a factory function that given a
scikit-learn estimator, returns a pipeline that combines this estimator with a
preprocessing steps. Those steps corresponds to a :class:`TableVectorizer` that
is in charge of dealing with heterogeneous data and depending on the capabilities of
the final estimator, a missing value imputer.

In the next section, we provide more details regarding the :class:`TableVectorizer`.

The parameters of the :class:`TableVectorizer` are chosen based on the type of the final
estimator.

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

With linear models or unknown models, the default values of the different parameters are
used. Therefore, the :obj:`GapEncoder` is used for high-cardinality categorical features
and the :obj:`~sklearn.preprocessing.OneHotEncoder` for low-cardinality ones. If the
final estimator does not support missing values, a :obj:`~sklearn.impute.SimpleImputer`
is added before the final estimator. Finally, a
:obj:`~sklearn.preprocessing.StandardScaler` is added to the pipeline. Those choices may
not be optimal in all cases but they are methodologically safe.

.. _table_vectorizer:

Turning a dataframe into a numerical feature matrix
---------------------------------------------------

A dataframe can contain columns of all kind of types. We usually refer such data to
"heterogeneous" data. A good numerical representation of these columns help analytics
and statistical learning.

The :class:`TableVectorizer` gives a turn-key solution by applying different
data-specific encoders to the different columns. It makes reasonable heuristic choices
that are not necessarily optimal since it is not aware of the learner used for the
machine learning task). However, it already provides a typically very good baseline.
