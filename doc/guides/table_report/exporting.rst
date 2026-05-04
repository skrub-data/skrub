.. |TableReport| replace:: :class:`~skrub.TableReport`
.. |set_config| replace:: :func:`~skrub.set_config`
.. |column_associations| replace:: :func:`~skrub.column_associations`

.. _user_guide_table_report_sharing:
How to export and share the |TableReport|
-----------------------------------------

The |TableReport| is generated as a standalone HTML file that includes the report
data, the plots, and the Javascript necessary to provide interactivity.

If it is generated inside a notebook (Jupyter or Marimo), the |TableReport| is
rendered directly inside the cell where it is called. If, instead, it is generated
by a script, the report will need to be opened by calling ``.open()``:

>>> TableReport(df).open() # doctest: +SKIP

Note that calling ``.open()`` will start a standalone process that hosts the report,
and a tab will be opened in the default browser. It is not possible to save the
report from the webpage. The function :func:`~skrub.TableReport.write_html` should
be used for that:

.. code-block::

    tr = TableReport(df)
    tr.write_html("my_report.html")

It is also possible to export the raw HTML, or a HTML fragment to embed in a page
with :func:`~skrub.TableReport.html` and  :func:`~skrub.TableReport.html_snippet`
respectively.

Finally, it is possible to export the data in JSON format, which allows structured
access to the data and statistics used to build the report with
:func:`~skrub.TableReport.json`.

.. code-block::

    tr = TableReport(df)
    json_data = tr.json()

Note that this will export all parts of the |TableReport|, including the distribution
plots in SVG format if they have been generated. If you do not need them, plots should be
disabled directly when generating the table report.

.. code-block::

    tr = TableReport(df, plot_distributions=False)
    json_data = tr.json()
