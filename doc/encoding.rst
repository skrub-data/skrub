.. _encoding:

====================================
Encoding: creating feature matrices
====================================

.. currentmodule:: skrub

Encoding or vectorizing creates numerical features from the data,
converting dataframes, strings, dates... Different encoders are suited
for different types of data.

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

.. topic:: References

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


Encoding text with diverse entries
----------------------------------

When it comes to encoding text with diverse entries, large language models,
fine-tuned for embedding purposes, improve data processing. Diverse entries are
typically represented by free-form text, i.e. strings with markedly more unique
ngrams than dirty categories.

Skrub provides :class:`SentenceEncoder`: a wrapper around
`SentenceTransformer <https://sbert.net/>`_ package that lets you use any embedding
model available on the HuggingFace Hub for Sentence Transformers.

.. topic:: References

    See `Vectorizing string entries for data processing on tables: when are larger
    language models better? <https://hal.science/hal-043459>`_ [3]_
    for a comparison between the dirty categories and diverse entries regimes against
    large language models and string-based encoders like the :class:`MinHashEncoder`.

.. [3]  L. Grinsztajn, M. Kim, E. Oyallon, G. Varoquaux.
        Vectorizing string entries for data processing on tables: when are larger
        language models better? 2023.


Encoding dates
---------------

The :class:`DatetimeEncoder` encodes date and time: it represent them as
time in seconds since a fixed date, but also added features useful to
capture regularities: week of the day, month of the year...
