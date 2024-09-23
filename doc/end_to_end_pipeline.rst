.. _end_to_end_pipeline:

============================
End-to-end predictive models
============================

.. currentmodule:: skrub

.. _tabular_learner:

Create baseline predictive models on heterogeneous dataset
----------------------------------------------------------

Crafting a machine-learning pipeline is a rather daunting task. Choosing the ending
learner of such pipeline is usually the easiest part. However, it imposes constraints
regarding the preprocessing steps that are are required ahead of the learner.
Programmatically defining these steps is the part that requires the most expertise and
that is cumbersome to write.

The function :func:`tabular_learner` provides a factory function that, given a
scikit-learn estimator, returns a pipeline combining this estimator with the
appropriate preprocessing steps. These steps correspond to a :class:`TableVectorizer`
that handles heterogeneous data and, depending on the capabilities
of the final estimator, a missing value imputer and/or a standard scaler.

In the next section, we provide more details regarding the :class:`TableVectorizer`
(:ref:`table_vectorizer`). The parameters of the :class:`TableVectorizer` are chosen
based on the type of the final estimator.

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

Turning a dataframe into a numeric feature matrix
-------------------------------------------------

A dataframe can contain columns of all kinds of types. We usually refer to such data as
"heterogeneous" data. A good numerical representation of these columns helps with
analytics and statistical learning.

The :class:`TableVectorizer` gives a turn-key solution by applying different
data-specific encoders to the different columns. It makes reasonable heuristic choices
that are not necessarily optimal since it is not aware of the learner used for the
machine learning task. However, it already provides a typically very good baseline.

The :class:`TableVectorizer` handles the following type of data:

- numerical data represented with the data types `bool`, `int`, and `float`;
- categorical data represented with the data types `str` or categorical (e.g.
  :obj:`pandas.CategoricalDtype` or :obj:`polars.datatypes.Categorical`);
- date and time data represented by `datetime` data type (e.g. :obj:`numpy.datetime64`,
  :obj:`pandas.DatetimeTZDtype`, :obj:`polars.datatypes.Datetime`).

Categorical data are subdivided into two groups: columns containing a large number of
categories (high-cardinality columns) and columns containing a small number of
categories (low-cardinality columns). A column is considered high-cardinality if the
number of unique values is greater than a given threshold, which is controlled by the
parameter cardinality_threshold.

Each group of data types defined earlier is associated with a specific init parameter
(e.g. ``numeric``, ``datetime``, etc.). The value of these parameters follows the same
convention:

- when set to ``"passthrough"``, the input columns are output as they are;
- when set to ``"drop"``, the input columns are dropped;
- when set to `a compatible scikit-learn transformer <https://scikit-learn.org/stable/glossary.html#term-transformer>`_
  (implementing ``fit``, ``transform``, and ``fit_transform`` methods), the transformer
  is applied to each column independently. The transformer is cloned (using
  :func:`sklearn.base.clone`) before calling the ``fit`` method.

Examples
--------

The following examples provide an in-depth look at how to use the
:class:`~skrub.TableVectorizer` class and the :func:`~skrub.tabular_learner`
function.

- :ref:`sphx_glr_auto_examples_01_encodings.py`
