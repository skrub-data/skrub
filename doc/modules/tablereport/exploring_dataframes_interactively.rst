.. |TableReport| replace:: :class:`~skrub.TableReport`
.. |set_config| replace:: :func:`~skrub.set_config`
.. |column_associations| replace:: :func:`~skrub.column_associations`

.. _user_guide_table_report_start:

Exploring dataframes interactively with the |TableReport|
=========================================================

The |TableReport| gives a high-level overview of a Dataframe or Series, suitable for
quick exploratory analysis. The report shows the first
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pre-computed examples of the |TableReport| are available
`here <https://skrub-data.org/skrub-reports/examples/index.html>`_, and you can
try it out on your data `here <https://skrub-data.org/skrub-reports/index.html>`_.

In the **Distributions** tab, it is possible to select columns by clicking on the
checkmark icon: the name of the column is added to the bar on top, so that it may
be copied in a script.

The TableReport can be used in a notebook cell, or it can be opened in a browser
window using ``TableReport(df).open()``.

.. _user_guide_table_report_customize:

Altering the Appearance of the |TableReport|
--------------------------------------------

The skrub global configuration includes various parameters that allow to tweak
the HTML representation of the |TableReport|.

For performance reasons, the |TableReport| disables the computation of
distributions and associations for tables with more than 30 columns. This behavior
can be changed by modifying the ``max_plot_columns`` and ``max_association_columns``
parameter.

It is also possible to specify the floating point precision by setting the appropriate
``float_precision`` parameter.

Parameters can be made permanent in a script by altering the configuration with
|set_config|, or by setting the respective environment variables. Refer to
:ref:`user_guide_configuration_parameters` for more detail.

.. _user_guide_table_report_sharing:

Exporting and Sharing the |TableReport|
---------------------------------------

The |TableReport| is a standalone object that does not require a running notebook
to be accessed after generation: it can be exported in HTML format and opened
directly in a browser as a HTML page.

>>> import io # to avoid writing to disk in the example
>>> from skrub import TableReport
>>> import pandas as pd
>>> df = pd.DataFrame({
...     "id": [1, 2, 3],
...     "value": [10, 20, 30],
... })
>>> tr = TableReport(df)
>>> html_buffer = io.StringIO()
>>> tr.write_html(html_buffer)  # save to file
>>> html = tr.html()  # get a string containing the HTML for a full page
>>> html_snippet = tr.html_snippet()  # get an HTML fragment to embed in a page
>>> tr_json = tr.json()  # get the content of the report in JSON format
