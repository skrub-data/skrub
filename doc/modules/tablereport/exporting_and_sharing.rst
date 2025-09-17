.. |TableReport| replace:: :class:`~skrub.TableReport`

Customizing and sharing the |TableReport|
=========================================

.. |set_config| replace:: :func:`~skrub.set_config`

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

.. |TableReport| replace:: :class:`~skrub.TableReport`

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
