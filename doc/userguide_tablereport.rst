.. _userguide_tablereport:
.. |TableReport| replace:: :class:`~skrub.TableReport`
.. |set_config| replace:: :func:`~skrub.set_config`
.. |column_associations| replace:: :func:`~skrub.column_associations`

=========================================

Using the |TableReport| to explore dataframes
---------------------------------------------

Exploring and Reporting DataFrames with the |TableReport|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from skrub import TableReport
    TableReport(df)  # from a notebook cell
    TableReport(df).open()  # to open in a browser window

The |TableReport| gives a high-level overview of the given dataframe, suitable for
quick exploratory analysis of series and dataframes. The report shows the first
and last 5 rows of the dataframe (decided by the ``n_rows`` parameter), as well
as additional information in other tabs.

- The **Stats** tab reports high-level statistics for each column.
- The **Distribution** tab collects summary plots for each column (max 30 by default).
- The **Associations** tab shows Cramer V and Pearson correlations between columns.
- Built-in filters allow selection of columns by dtype and other conditions.

In the **Distributions** tab, it is possible to select columns by clicking on the
checkmark icon: the name of the column is added to the bar on top, so that it may
be copied in a script.

Altering the Appearance of the |TableReport|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For performance reasons, the |TableReport| disables the computation of
Distributions and Associations for tables with more than 30 columns. This behavior
can be changed by modifying the ``max_plot_columns`` and ``max_association_columns``
parameter, or by altering the configuration with |set_config| (refer to the
|TableReport| and |set_config| docs for more detail).

More pre-computed examples are available `here <https://skrub-data.org/skrub-reports/examples/index.html>`_.

Exporting and Sharing the |TableReport|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The |TableReport| is a standalone object that does not require a running notebook
to be accessed after generation: it can be exported in HTML format and opened
directly in a browser as a HTML page.

.. code-block:: python

    tr = TableReport(df)
    tr.write_html("report.html")  # save to file
    tr.html()  # get a string containing the HTML for a full page
    tr.html_snippet()  # get an HTML fragment to embed in a page
    tr.json()  # get the content of the report in JSON format

Finding Correlated Columns in a DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The |TableReport|'s **Associations** tab shows this information. It is also
possible to use the |column_associations| function, which returns a dataframe
containing the associations.
