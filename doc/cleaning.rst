.. _cleaning:

=========
Cleaning
=========

.. currentmodule:: skrub

:func:`deduplicate`: merging variants of the same entry
=========================================================

:func:`deduplicate` is used to merge multiple variants of the same entry
into one, for instance typos. Such cleaning is needed to apply subsequent
dataframe operations that need exact correspondances, such as counting
elements. It is typically not needed when the entries are fed directly to
a machine-learning model, as a :ref:`dirty-category encoder <dirty_categories>`
can suffice.
