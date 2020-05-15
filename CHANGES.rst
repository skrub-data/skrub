Release 0.0.7
=============
* **datasets.fetch_employee_salaries**: change the origin of download for employee_salaries.
  - The function now return a bunch with a dataframe under the field "data",
    and not the path to the csv file. 
  - The field "description" has been renamed to "DESCR".


Release 0.0.6
=============
* **SimilarityEncoder**: Fixed a bug when using the Jaro-Winkler distance as a
  similarity metric. Our implementation now accurately reproduces the behaviour
  of the ``python-Levenshtein`` implementation.

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
