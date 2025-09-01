.. |DropUninformative| replace:: :class:`~skrub.DropUninformative`
.. |Cleaner| replace:: :class:`~skrub.Cleaner`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |deduplicate| replace:: :func:`~skrub.deduplicate`
.. |SquashingScaler| replace:: :class:`~skrub.SquashingScaler`
.. |RobustScaler| replace:: :class:`~sklearn.preprocessing.RobustScaler`
.. |SquashingScaler| replace:: :class:`~skrub.SquashingScaler`
.. |RobustScaler| replace:: :class:`~sklearn.preprocessing.RobustScaler`

Deduplicate categorical data with |deduplicate|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a series containing strings with typos, the |deduplicate| function
may be used to remove some typos by creating a mapping between the typo strings
and the correct strings. See the documentation for caveats and more detail.
