.. _cleaning:

=========
Cleaning
=========

.. currentmodule:: skrub

:class:`Cleaner`: sanitizing a dataframe
=========================================

:class:`Cleaner` sanitizes a dataframe, transforming it to a more
consistent data representation which is easier to work with: it detects
null values represented as strings, parses dates, and removes
uninformative columns (see docstring: :class:`Cleaner`).

.. Code in a comment to build the variables we need for doctests to run

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': ['one', 'two', 'two', 'three'],
    ...     'B': ['02/02/2024', '23/02/2024', '12/03/2024', '13/03/2024'],
    ...     'C': ['1.5', 'N/A', '12.2', 'N/A'],
    ...     'D': [1.5, 2.0, 2.5, 3.0],
    ... })
    >>> df_test = df

To have reproducible transformations, it is implemented as a scikit-learn
transformer:

* to sanitize a given dataframe `df`::

    >>> from skrub import Cleaner
    >>> cleaner = Cleaner(drop_if_constant=True)
    >>> clean_df = cleaner.fit_transform(df)

* to apply the same exact operations to a new dataframe `df_test`
  (new rows with the same columns)::

   >>> clean_df_test = cleaner.transform(df_test)

  Reusing the cleaner to transform new data ensures that if columns were
  dropped on the first dataframe, they are dropped on the second.

|

:func:`deduplicate`: merging variants of the same entry
=========================================================

:func:`deduplicate` is used to merge multiple variants of the same entry
into one, for instance typos. Such cleaning is needed to apply subsequent
dataframe operations that need exact correspondences, such as counting
elements. It is typically not needed when the entries are fed directly to
a machine-learning model, as a :ref:`dirty-category encoder <dirty_categories>`
can suffice.
