import functools
import json
from pathlib import Path

from ._html import to_html
from ._serve import open_file_in_browser, open_in_browser
from ._summarize import summarize_dataframe
from ._utils import JSONEncoder


class Report:
    """Summarize the contents of a dataframe.

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
    """

    def __init__(self, dataframe, order_by=None, title=None, column_filters=None):
        self._summary_kwargs = {"order_by": order_by}
        self.title = title
        self.column_filters = column_filters
        self.dataframe = dataframe

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

    @functools.cached_property
    def html(self):
        return to_html(
            self._summary_with_plots,
            standalone=True,
            column_filters=self.column_filters,
        )

    @functools.cached_property
    def html_snippet(self):
        return to_html(
            self._summary_with_plots,
            standalone=False,
            column_filters=self.column_filters,
        )

    @functools.cached_property
    def json(self):
        to_remove = ["dataframe", "head", "tail", "first_row_dict"]
        data = {
            k: v for k, v in self._summary_without_plots.items() if k not in to_remove
        }
        return json.dumps(data, cls=JSONEncoder)

    def _repr_mimebundle_(self, include=None, exclude=None):
        del include, exclude
        return {"text/html": self.html_snippet}

    def _repr_html_(self):
        return self._repr_mimebundle_()["text/html"]

    def open(self, file_path=None):
        """Open the HTML report in a web browser.

        Parameters
        ----------
        file_path : str or pathlib.Path
            If provided, the report is saved at the specified location and the
            file is opened in the browser. If ``None``, nothing is written to
            disk. A server is started to send the report to the browser and
            shut down immediately afterwards; refreshing the page will result
            in a "Not found" error.
        """
        if file_path is None:
            open_in_browser(self.html)
            return
        file_path = Path(file_path).resolve()
        file_path.write_text(self.html, "UTF-8")
        open_file_in_browser(file_path)