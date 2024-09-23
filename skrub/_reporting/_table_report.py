import functools
import json

from ._html import to_html
from ._serve import open_in_browser
from ._summarize import summarize_dataframe
from ._utils import JSONEncoder


class TableReport:
    r"""Summarize the contents of a dataframe.

    This class summarizes a dataframe, providing information such as the type
    and summary statistics (mean, number of missing values, etc.) for each
    column.

    Parameters
    ----------
    dataframe : pandas or polars DataFrame
        The dataframe to summarize.
    n_rows : int, default=10
        Maximum number of rows to show in the sample table. Half will be taken
        from the beginning (head) of the dataframe and half from the end
        (tail). Note this is only for display. Summary statistics, histograms
        etc. are computed using the whole dataframe.
    order_by : str
        Column name to use for sorting. Other numerical columns will be plotted
        as function of the sorting column. Must be of numerical or datetime
        type.
    title : str
        Title for the report.
    column_filters : dict
        A dict for adding custom entries to the column filter dropdown menu.
        Each key is an id for the filter (e.g. ``"first_10"``) and the value is a
        mapping with the keys ``display_name`` (the name shown in the menu,
        e.g. ``"First 10 columns"``) and ``columns`` (a list of column names).
        See the end of the "Examples" section below for details.

    Notes
    -----
    You can see some `example reports`_ for a few datasets online. We also
    provide an experimental online demo_ that allows you to select a CSV or
    parquet file and generate a report directly in your web browser.

    .. _example reports: https://skrub-data.org/skrub-reports/examples/
    .. _demo: https://skrub-data.org/skrub-reports/

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub import TableReport
    >>> df = pd.DataFrame(dict(a=[1, 2], b=['one', 'two'], c=[11.1, 11.1]))
    >>> report = TableReport(df)

    If you are in a Jupyter notebook, to display the report just have it be the
    last expression evaluated in a cell so that it is displayed in the cell's
    output.

    >>> report
    <TableReport: use .open() to display>

    (Note that above we only see the string represention, not the report itself,
    because we are not in a notebook.)

    Whether you are using a notebook or not, you can always open the report as a
    full page in a separate browser tab with its ``open`` method:
    ``report.open()``.

    You can also get the HTML report as a string.
    For a full, standalone web page:

    >>> report.html()
    Processing...
    '<!DOCTYPE html>\n<html lang="en-US">\n\n<head>\n    <meta charset="utf-8"...'

    For an HTML fragment that can be inserted into a page:

    >>> report.html_snippet()
    '\n<div id="report_...-wrapper" hidden>\n    <template id="report_...'

    Advanced configuration: you can add custom column filters that will appear
    in the report's dropdown menu.

    >>> filters = {
    ...     "at_least_2": {
    ...         "display_name": "Columns with at least 2 unique values",
    ...         "columns": ["a", "b"],
    ...     }
    ... }
    >>> report = TableReport(df, column_filters=filters)

    With the code above, in addition to the default filters such as "All
    columns", "Numeric columns", etc., the added "Columns with at least 2
    unique values" will be available in the report, selecting columns "a" and
    "b".
    """

    def __init__(
        self,
        dataframe,
        n_rows=10,
        order_by=None,
        title=None,
        column_filters=None,
    ):
        n_rows = max(1, n_rows)
        self._summary_kwargs = {
            "order_by": order_by,
            "max_top_slice_size": -(n_rows // -2),
            "max_bottom_slice_size": n_rows // 2,
        }
        self.title = title
        self.column_filters = column_filters
        self.dataframe = dataframe

    def __repr__(self):
        return f"<{self.__class__.__name__}: use .open() to display>"

    @functools.cached_property
    def _summary_with_plots(self):
        return summarize_dataframe(
            self.dataframe, with_plots=True, title=self.title, **self._summary_kwargs
        )

    @functools.cached_property
    def _summary_without_plots(self):
        return summarize_dataframe(
            self.dataframe, with_plots=False, title=self.title, **self._summary_kwargs
        )

    @property
    def _any_summary(self):
        if "_summary_with_plots" in self.__dict__:
            return self._summary_with_plots
        return self._summary_without_plots

    def html(self):
        """Get the report as a full HTML page.

        Returns
        -------
        str :
            The HTML page.
        """
        return to_html(
            self._summary_with_plots,
            standalone=True,
            column_filters=self.column_filters,
        )

    def html_snippet(self):
        """Get the report as an HTML fragment that can be inserted in a page.

        Returns
        -------
        str :
            The HTML snippet.
        """
        return to_html(
            self._summary_with_plots,
            standalone=False,
            column_filters=self.column_filters,
        )

    def json(self):
        """Get the report data in JSON format.

        Returns
        -------
        str :
            The JSON data.
        """
        to_remove = ["dataframe", "sample_table", "first_row_dict"]
        data = {
            k: v for k, v in self._summary_without_plots.items() if k not in to_remove
        }
        return json.dumps(data, cls=JSONEncoder)

    def _repr_mimebundle_(self, include=None, exclude=None):
        del include, exclude
        return {"text/html": self.html_snippet()}

    def _repr_html_(self):
        return self._repr_mimebundle_()["text/html"]

    def open(self):
        """Open the HTML report in a web browser."""
        open_in_browser(self.html())
