.. |StringEncoder| replace:: :class:`~skrub.StringEncoder`
.. |TextEncoder| replace:: :class:`~skrub.TextEncoder`
.. |MinHashEncoder| replace:: :class:`~skrub.MinHashEncoder`
.. |GapEncoder| replace:: :class:`~skrub.GapEncoder`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |OneHotEncoder| replace:: :class:`~sklearn.preprocessing.OneHotEncoder`
.. |OrdinalEncoder| replace:: :class:`~sklearn.preprocessing.OrdinalEncoder`

|StringEncoder|: the default encoder, strong in most cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A strong and quick baseline for both short strings with high cardinality and long
text. This encoder computes the ngram frequency using tf-idf vectorization,
followed by truncated SVD
(`Latent Semantic Analysis <https://en.wikipedia.org/wiki/Latent_semantic_analysis>`_).
