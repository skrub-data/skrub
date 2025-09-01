.. |DropUninformative| replace:: :class:`~skrub.DropUninformative`
.. |Cleaner| replace:: :class:`~skrub.Cleaner`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |deduplicate| replace:: :func:`~skrub.deduplicate`
.. |SquashingScaler| replace:: :class:`~skrub.SquashingScaler`
.. |RobustScaler| replace:: :class:`~sklearn.preprocessing.RobustScaler`
.. |SquashingScaler| replace:: :class:`~skrub.SquashingScaler`
.. |RobustScaler| replace:: :class:`~sklearn.preprocessing.RobustScaler`

Removing unneeded columns with |DropUninformative| and |Cleaner|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|DropUninformative| is used to remove features or data points that do not provide
useful information for the analysis or model.

Tables may include columns that do not carry useful information. These columns
increase computational cost and may reduce downstream performance.

The |DropUninformative| transformer includes various heuristics to drop columns
considered "uninformative":

- Drops all columns that contain only missing values (threshold adjustable via
  ``drop_null_fraction``)
- Drops columns with only a single value if ``drop_if_constant=True``
- Drops string/categorical columns where each row is unique if
  ``drop_if_unique=True`` (use with care)

|DropUninformative| is used by both |TableVectorizer| and |Cleaner|; both accept
the same parameters to drop columns accordingly.
