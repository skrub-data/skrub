.. |TableReport| replace:: :class:`~skrub.TableReport`
.. |column_associations| replace:: :func:`~skrub.column_associations`

.. _user_guide_table_report_associations:

Finding Correlated Columns in a DataFrame
=========================================

In addition to |TableReport|'s **Associations** tab, you can compute associations
using the |column_associations| function, which returns a dataframe containing the
associations.

Reported metrics include `Cramer’s V statistic <https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V>`_
and `Pearson’s Correlation Coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_.
The result is returned as a dataframe that contains the column name and idx for the
left and right table and both associations; results are sorted in descending order
by Cramer’s V association.

This can be useful to have access to the information used in the |TableReport|
for later use (e.g., to select which columns to drop).
