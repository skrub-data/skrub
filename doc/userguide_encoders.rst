.. |StringEncoder| replace:: :class:`~skrub.StringEncoder`
.. |TextEncoder| replace:: :class:`~skrub.TextEncoder`
.. |MinHashEncoder| replace:: :class:`~skrub.MinHashEncoder`
.. |GapEncoder| replace:: :class:`~skrub.GapEncoder`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |OneHotEncoder| replace:: :class:`~sklearn.preprocessing.OneHotEncoder`
.. |OrdinalEncoder| replace:: :class:`~sklearn.preprocessing.OrdinalEncoder`

.. _userguide_encoders:

Feature engineering for categorical data
--------------------------------------------------

In Skrub, categorical features correspond to columns whose data type is neither numeric nor
datetime. This includes string, categorical, and object data types.


|StringEncoder|
~~~~~~~~~~~~~~

A strong and quick baseline for both short strings with high cardinality and long
text. This encoder computes the ngram frequency using tf-idf vectorization,
followed by truncated SVD
(`Latent Semantic Analysis <https://en.wikipedia.org/wiki/Latent_semantic_analysis>`_).

|TextEncoder|
~~~~~~~~~~~~~

This encoder encodes string features using pretrained language models from the
HuggingFace Hub. It is a wrapper around `sentence-transformers <https://sbert.net/>`_
compatible with the scikit-learn API and usable in pipelines. Best for
free-flowing text and when columns include context found in the pretrained model
(e.g., name of cities etc.). Note that this encoder can take a very long time to
train, especially on large datasets and on CPU.

|MinHashEncoder|
~~~~~~~~~~~~~~~~

This encoder decomposes strings into ngrams, then applies the MinHash method to convert them
into numerical features. Fast to train, but features may yield worse results
compared to other methods.

|GapEncoder|
~~~~~~~~~~~~

The |GapEncoder| estimates "latent categories" on the training data by finding
common ngrams between strings, then encodes the categories as real
numbers. It allows access to grouped features via ``.get_feature_names_out()``,
which allows for better interpretability. This encoder may require a long time to train.

Comparison of the Categorical Encoders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 15 15 25 20 25

    * - Encoder
      - Training time
      - Performance on categorical data
      - Performance on text data
      - Notes
    * - StringEncoder
      - Fast
      - Good
      - Good
      -
    * - TextEncoder
      - Very slow
      - Mediocre to good
      - Very good
      - Requires the ``transformers`` dep.
    * - GapEncoder
      - Slow
      - Good
      - Mediocre to good
      - Interpretable
    * - MinHashEncoder
      - Very fast
      - Mediocre to good
      - Mediocre
      -

:ref:`This example <example_string_encoders>`) and this `blog post <https://skrub-data.org/skrub-materials/pages/notebooks/categorical-encoders/categorical-encoders.html>`_ include a more systematic analysis of each method.
