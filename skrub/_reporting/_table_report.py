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
    order_by : str
        Column name to use for sorting. Other numerical columns will be plotted
        as function of the sorting column. Must be of numerical or datetime
        type.
    title : str
        Title for the report.
    column_filters : dict
        A dict for adding custom entries to the column filter dropdown menu.
        Each key is an id for the filter (e.g. ``"all()"``) and the value is a
        mapping with the keys ``display_name`` (the name shown in the menu,
        e.g. ``"All columns"``) and ``columns`` (a list of column names).

    Attributes
    ----------
    html : str
        Report as an HTML page.
    html_snippet : str
        Report as an HTML snippet containing a single '<div>' element. Useful
        to embed the report in an HTML page or displaying it in a Jupyter
        notebook.
    json : str
        Report in JSON format.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub import TableReport
    >>> df = pd.DataFrame(dict(a=[1, 2], b=['one', 'two']))
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
    '<!DOCTYPE html>\n<html lang="en-US">\n\n<head>\n    <meta charset="utf-8"...'

    For an HTML fragment that can be inserted into a page:

    >>> report.html_snippet()
    '\n<div id="report_...-wrapper" hidden>\n    <template id="report_...'
    """

    def __init__(self, dataframe, order_by=None, title=None, column_filters=None):
        self._summary_kwargs = {"order_by": order_by}
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
        to_remove = ["dataframe", "head", "tail", "first_row_dict"]
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
