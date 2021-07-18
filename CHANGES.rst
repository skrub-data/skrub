Release 0.2.0
=============

Version 0.2.0 is not yet releave. To install it use::

    pip install git+https://github.com/dirty-cat/dirty_cat.git

Major changes
-------------

* **PretrainedFastText**: Added encoding of categorical variables through
  a pretrained fastText model. FastText word embeddings capture both semantic
  and morphological similarities.
  
* Bump minimum dependencies:

  - Python (>= 3.6)
  - NumPy (>= 1.16)
  - SciPy (>= 1.2)
  - scikit-learn (>= 0.20.0)

* **SuperVectorizer**: Added automatic transform through the
  :class:`SuperVectorizer` class. It transforms
  columns automatically based on their type. It provides a replacement
  for scikit-learn's `ColumnTransformer` simpler to use on heterogeneous
  panda dataframes.


Bug-fixes
---------

* Fix get_feature_names under scikit-learn > 0.21


Release 0.1.1
=============

Major changes
-------------
  
Bug-fixes
---------

* RuntimeWarnings due to overflow in GapEncoder (#161)


Release 0.1.0
=============

Major changes
-------------

* **GapEncoder**: Added online Gamma-Poisson factorization through the
  :class:`GapEncoder` class. This method discovers latent categories formed
  via combinations of substrings, and encodes string data as combinations of
  these categories. To be used if interpretability is important.

Bug-fixes
---------

* Multiprocessing exception in notebook (#154)


Release 0.0.7
=============

* **MinHashEncoder**: Added ``minhash_encoder.py`` and ``fast_hast.py`` files
  that implement minhash encoding through the ``MinHashEncoder`` class.
  This method allows for fast and scalable encoding of string categorical
  variables.

* **datasets.fetch_employee_salaries**: change the origin of download for employee_salaries.

  - The function now return a bunch with a dataframe under the field "data",
    and not the path to the csv file. 
  - The field "description" has been renamed to "DESCR".

* **SimilarityEncoder**: Fixed a bug when using the Jaro-Winkler distance as a
  similarity metric. Our implementation now accurately reproduces the behaviour
  of the ``python-Levenshtein`` implementation.

* **SimilarityEncoder**: Added a "handle_missing" attribute to allow encoding
  with missing values.

* **TargetEncoder**: Added a "handle_missing" attribute to allow encoding
  with missing values.

* **MinHashEncoder**: Added a "handle_missing" attribute to allow encoding
  with missing values.

Release 0.0.6
=============

* **SimilarityEncoder**: Accelerate ``SimilarityEncoder.transform``, by:

  - computing the vocabulary count vectors in ``fit`` instead of ``transform``
  - computing the similarities in parallel using ``joblib``. This option can be
    turned on/off via the ``n_jobs`` attribute of the ``SimilarityEncoder``.

* **SimilarityEncoder**: Fix a bug that was preventing a ``SimilarityEncoder``
  to be created when ``categories`` was a list.

* **SimilarityEncoder**: Set the dtype passed to the ngram similarity
  to float32, which reduces memory consumption during encoding.

Release 0.0.5
=============

* **SimilarityEncoder**: Change the default ngram range to (2, 4) which
  performs better empirically.

* **SimilarityEncoder**: Added a "most_frequent" strategy to define
  prototype categories for large-scale learning.

* **SimilarityEncoder**: Added a "k-means" strategy to define prototype
  categories for large-scale learning.

* **SimilarityEncoder**: Added the possibility to use hashing ngrams for
  stateless fitting with the ngram similarity.

* **SimilarityEncoder**: Performance improvements in the ngram similarity.

* **SimilarityEncoder**: Expose a get_feature_names method.
