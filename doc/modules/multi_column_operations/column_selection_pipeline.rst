Column selection inside a pipeline
----------------------------------

Besides joins, another common operation on a dataframe is to select a subset of its columns (also known as a projection).
We sometimes need to perform such a selection in the middle of a pipeline, for example if we need a column for a join (with :class:`Joiner`), but in a subsequent step we want to drop that column before fitting an estimator.

skrub provides transformers to perform such an operation:

- :class:`SelectCols` allows specifying the columns we want to keep.
- Conversely :class:`DropCols` allows specifying the columns we want to discard.
