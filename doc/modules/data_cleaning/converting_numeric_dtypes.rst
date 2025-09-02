.. |DropUninformative| replace:: :class:`~skrub.DropUninformative`
.. |Cleaner| replace:: :class:`~skrub.Cleaner`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |deduplicate| replace:: :func:`~skrub.deduplicate`
.. |SquashingScaler| replace:: :class:`~skrub.SquashingScaler`
.. |RobustScaler| replace:: :class:`~sklearn.preprocessing.RobustScaler`
.. |SquashingScaler| replace:: :class:`~skrub.SquashingScaler`
.. |RobustScaler| replace:: :class:`~sklearn.preprocessing.RobustScaler`

Converting numeric dtypes to ``float32`` with the |Cleaner|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the |Cleaner| tries to parse numeric datatypes even if  and does not cast them to a
different dtype. In some cases, it may be beneficial to have the same numeric
dtype for all numeric columns to guarantee compatibility between values.

The |Cleaner| allows conversion of numeric features to ``float32`` by setting
the ``numeric_dtype`` parameter:

>>> from skrub import Cleaner
>>> cleaner = Cleaner(numeric_dtype="float32")

Setting the dtype to ``float32`` reduces RAM footprint for most use cases and
ensures that all missing values have the same representation. This also ensures
compatibility with scikit-learn transformers.
