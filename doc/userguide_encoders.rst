.. _userguide_encoders
.. |StringEncoder| replace:: :class:`~skrub.StringEncoder`
.. |TextEncoder| replace:: :class:`~skrub.TextEncoder`
.. |MinHashEncoder| replace:: :class:`~skrub.MinHashEncoder`
.. |GapEncoder| replace:: :class:`~skrub.GapEncoder`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |OneHotEncoder| replace:: :class:`~sklearn.preprocessing.OneHotEncoder`
.. |OrdinalEncoder| replace:: :class:`~sklearn.preprocessing.OrdinalEncoder`

Encoding String and Text Data as Numerical Features
--------------------------------------------------

In ``skrub``, categorical features are all features not detected as numeric or datetimes: this includes strings, text, IDs, and features with dtype ``categorical`` (e.g., ``pd.Categorical``).

High Cardinality and Low Cardinality Categorical Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In machine learning pipelines, these features are converted to numerical features
using various encodings (|OneHotEncoder|, |OrdinalEncoder|, etc.). Typically,
categorical features are encoded using |OneHotEncoder|, but this can cause issues
when the number of unique values (the "cardinality") is very large.

The |TableVectorizer| classifies categorical features with more than 40 unique
values as *high cardinality*, and all others as *low cardinality*. Different
encoding strategies are applied to each kind; the threshold can be modified with
the ``cardinality_threshold`` parameter.

- Low cardinality: encoded by default using scikit-learn |OneHotEncoder|
- High cardinality: encoded using the |StringEncoder|

Categorical encoding is applied only to columns that do not have a string or categorical dtype.

|StringEncoder|
~~~~~~~~~~~~~~

A strong and quick baseline for both short strings with high cardinality and long
text. Applies tf-idf vectorization followed by truncated SVD
(`Latent Semantic Analysis <https://en.wikipedia.org/wiki/Latent_semantic_analysis>`_).

|TextEncoder|
~~~~~~~~~~~~~

Encodes string features using pretrained models from the HuggingFace Hub. It is a
wrapper around ``SentenceTransformer`` compatible with the scikit-learn API and
usable in pipelines. Best for free-flowing text and when columns include context
found in the pretrained model.

|MinHashEncoder|
~~~~~~~~~~~~~~~~

Decomposes strings into ngrams, then applies the MinHash method to convert them
into numerical features. Fast to train, but features may yield worse results
compared to other methods.

|GapEncoder|
~~~~~~~~~~~~

Estimates "latent categories" on the training data, then encodes them as real
numbers. Allows access to grouped features via ``.get_feature_names_out()``. May
require a long time to train.

Comparison of the Categorical Encoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+------------------+---------------+-------------------------------+------------------------+--------------------------------------+
|     Encoder      | Training time | Performance on categorical     | Performance on text    | Notes                                |
|                  |               | data                          | data                   |                                      |
+==================+===============+===============================+========================+======================================+
| StringEncoder    | Fast          | Good                          | Good                   |                                      |
+------------------+---------------+-------------------------------+------------------------+--------------------------------------+
| TextEncoder      | Very slow     | Mediocre to good              | Very good              | Requires the ``transformers`` dep.   |
+------------------+---------------+-------------------------------+------------------------+--------------------------------------+
| GapEncoder       | Slow          | Good                          | Mediocre to good       | Interpretable                        |
+------------------+---------------+-------------------------------+------------------------+--------------------------------------+
| MinHashEncoder   | Very fast     | Mediocre to good              | Mediocre               |                                      |
+------------------+---------------+-------------------------------+------------------------+--------------------------------------+

Example 2 (see :ref:`_example_string_encoders`) and this `blog post <https://skrub-data.org/skrub-materials/pages/notebooks/categorical-encoders/categorical-encoders.html>`_ include a more systematic analysis of each method.
