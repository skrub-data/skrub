.. |TableReport| replace:: :class:`~skrub.TableReport`
.. |set_config| replace:: :func:`~skrub.set_config`
.. |column_associations| replace:: :func:`~skrub.column_associations`

Exploring dataframes interactively with the |TableReport|
=========================================================

The |TableReport| gives a high-level overview of the given dataframe, suitable for
quick exploratory analysis of series and dataframes. The report shows the first
and last 5 rows of the dataframe (decided by the ``n_rows`` parameter), as well
as additional information in other tabs.

- The **Stats** tab reports high-level statistics for each column.
- The **Distribution** tab collects summary plots for each column (max 30 by default).
- The **Associations** tab shows `Cramer V <https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V>`_
  and `Pearson correlation <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_
  between columns.
- Built-in filters allow selection of columns by dtype and other conditions.


The |TableReport| of a table can be generated as follows:
>>> from skrub import TableReport
>>> import pandas as pd
>>> df = pd.DataFrame({
...     "id": [1, 2, 3],
...     "value": [10, 20, 30],
... })
>>> TableReport(df)  # from a notebook cell
<TableReport: use .open() to display>

The command ``TableReport(df).open()`` opens the report in a browser window.

A demo of the |TableReport|
Pre-computed examples of of the |TableReport| are available
`here <https://skrub-data.org/skrub-reports/examples/index.html>`_, and you can
try it out on your data `here <https://skrub-data.org/skrub-reports/index.html>`_.

In the **Distributions** tab, it is possible to select columns by clicking on the
checkmark icon: the name of the column is added to the bar on top, so that it may
be copied in a script.

The TableReport can be used in a notebook cell, or it can be opened in a browser
window using ``TableReport(df).open()``.

It is also possible to export the |TableReport| in HTML or JSON format.
