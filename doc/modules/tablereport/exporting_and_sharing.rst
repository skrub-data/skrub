.. |TableReport| replace:: :class:`~skrub.TableReport`

Exporting and Sharing the |TableReport|
=======================================

The |TableReport| is a standalone object that does not require a running notebook
to be accessed after generation: it can be exported in HTML format and opened
directly in a browser as a HTML page.

>>> import io # to avoid writing to disk in the example
>>> tr = TableReport(df)
>>> html_buffer = io.StringIO()
>>> tr.write_html(html_buffer)  # save to file
>>> html = tr.html()  # get a string containing the HTML for a full page
>>> html_snippet = tr.html_snippet()  # get an HTML fragment to embed in a page
>>> tr_json = tr.json()  # get the content of the report in JSON format
