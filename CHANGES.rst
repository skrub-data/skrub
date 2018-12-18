Release 0.0.6
=============

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
