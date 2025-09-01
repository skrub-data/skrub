.. |StringEncoder| replace:: :class:`~skrub.StringEncoder`
.. |TextEncoder| replace:: :class:`~skrub.TextEncoder`
.. |MinHashEncoder| replace:: :class:`~skrub.MinHashEncoder`
.. |GapEncoder| replace:: :class:`~skrub.GapEncoder`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |OneHotEncoder| replace:: :class:`~sklearn.preprocessing.OneHotEncoder`
.. |OrdinalEncoder| replace:: :class:`~sklearn.preprocessing.OrdinalEncoder`

|TextEncoder|: language model-based, strong on text but expensive to run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This encoder encodes string features using pretrained language models from the
HuggingFace Hub. It is a wrapper around `sentence-transformers <https://sbert.net/>`_
compatible with the scikit-learn API and usable in pipelines. Best for
free-flowing text and when columns include context found in the pretrained model
(e.g., name of cities etc.). Note that this encoder can take a very long time to
train, especially on large datasets and on CPU.
