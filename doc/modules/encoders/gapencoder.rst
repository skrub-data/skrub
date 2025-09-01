.. |StringEncoder| replace:: :class:`~skrub.StringEncoder`
.. |TextEncoder| replace:: :class:`~skrub.TextEncoder`
.. |MinHashEncoder| replace:: :class:`~skrub.MinHashEncoder`
.. |GapEncoder| replace:: :class:`~skrub.GapEncoder`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |OneHotEncoder| replace:: :class:`~sklearn.preprocessing.OneHotEncoder`
.. |OrdinalEncoder| replace:: :class:`~sklearn.preprocessing.OrdinalEncoder`

|GapEncoder|: an interpretable, if slower encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The |GapEncoder| estimates "latent categories" on the training data by finding
common ngrams between strings, then encodes the categories as real
numbers. It allows access to grouped features via ``.get_feature_names_out()``,
which allows for better interpretability. This encoder may require a long time to train.
