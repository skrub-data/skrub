import codecs
import functools
import json
import numbers
from pathlib import Path

from .. import _config
from .. import _dataframe as sbd
from ._html import to_html
from ._serve import open_in_browser
from ._summarize import summarize_dataframe
from ._utils import JSONEncoder


def _check_max_cols(max_plot_columns, max_association_columns):
    max_plot_columns = (
        max_plot_columns
        if max_plot_columns is not None
        else _config.get_config()["max_plot_columns"]
    )
    if (max_plot_columns != "all") and not (
        isinstance(max_plot_columns, numbers.Real) and max_plot_columns >= 0
    ):
        raise ValueError(
            "'max_plot_columns' must be a positive scalar or 'all', got"
            f" {max_plot_columns!r}."
        )

    max_association_columns = (
        max_association_columns
        if max_association_columns is not None
        else _config.get_config()["max_association_columns"]
    )
    if (max_association_columns != "all") and not (
        isinstance(max_association_columns, numbers.Real)
        and max_association_columns >= 0
    ):
        raise ValueError(
            "'max_association_columns' must be a positive scalar or 'all', got "
            f"{max_association_columns!r}."
        )

    return max_plot_columns, max_association_columns


class TableReport:
    r"""Summarize the contents of a dataframe.

    This class summarizes a dataframe, providing information such as the type
    and summary statistics (mean, number of missing values, etc.) for each
    column.

    Parameters
    ----------
    dataframe : pandas or polars Series or DataFrame
        The dataframe or series to summarize.
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
    verbose : int, default = 1
        Whether to print progress information while the report is being generated.

        * verbose = 1 prints how many columns have been processed so far.
        * verbose = 0 silences the output.
    max_plot_columns : int, default=30
        Maximum number of columns for which plots should be generated.
        If the number of columns in the dataframe is greater than this value,
        the plots will not be generated. If "all", all columns will be plotted.

        To avoid having to set this parameter at each call of ``TableReport``, you can
        change the default using :func:`set_config`:

        >>> from skrub import set_config
        >>> set_config(max_plot_columns=30)

        You can also enable this default more permanently via an environment variable:

        .. code:: shell

            export SKB_MAX_PLOT_COLUMNS=30

    max_association_columns : int, default=30
        Maximum number of columns for which associations should be computed.
        If the number of columns in the dataframe is greater than this value,
        the associations will not be computed. If "all", the associations
        for all columns will be computed.

        To avoid having to set this parameter at each call of ``TableReport``, you can
        change the default using :func:`set_config`:

        >>> from skrub import set_config
        >>> set_config(max_association_columns=30)

        You can also enable this default more permanently via an environment variable:

        .. code:: shell

            export SKB_MAX_ASSOCIATION_COLUMNS=30

    See Also
    --------
    patch_display :
        Replace the default DataFrame HTML displays in the output of notebook
        cells with a TableReport.

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

    (Note that above we only see the string representation, not the report itself,
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
        verbose=1,
        max_plot_columns=None,
        max_association_columns=None,
    ):
        n_rows = max(1, n_rows)
        self._summary_kwargs = {
            "order_by": order_by,
            "max_top_slice_size": -(n_rows // -2),
            "max_bottom_slice_size": n_rows // 2,
            "verbose": verbose,
        }
        self._to_html_kwargs = {}
        self.title = title
        self.column_filters = column_filters
        self.verbose = verbose
        self.max_plot_columns, self.max_association_columns = _check_max_cols(
            max_plot_columns, max_association_columns
        )
        self.dataframe = (
            sbd.to_frame(dataframe) if sbd.is_column(dataframe) else dataframe
        )
        self.n_columns = sbd.shape(self.dataframe)[1]

    def _set_minimal_mode(self):
        """Put the report in minimal mode.

        This is meant to be called by other skrub functions, such as the
        DataOps  ``__repr__``.

        In the minimal mode, the associations and distributions tabs are not
        shown and the plots and associations are not computed.

        Once set this cannot be undone.
        """
        try:
            # delete the cached _summary if it already exists,
            # as the summarize arguments have changed
            delattr(self, "_summary")
        except AttributeError:
            pass
        self._to_html_kwargs["minimal_report_mode"] = True
        self.max_association_columns = 0
        self.max_plot_columns = 0

    def _display_subsample_hint(self):
        self._summary["is_subsampled"] = True

    def __repr__(self):
        return f"<{self.__class__.__name__}: use .open() to display>"

    @functools.cached_property
    def _summary(self):
        with_plots = (
            self.max_plot_columns == "all" or self.max_plot_columns >= self.n_columns
        )
        with_associations = (
            self.max_association_columns == "all"
            or self.max_association_columns >= self.n_columns
        )

        return summarize_dataframe(
            self.dataframe,
            with_plots=with_plots,
            with_associations=with_associations,
            title=self.title,
            **self._summary_kwargs,
        )

    def html(self):
        """Get the report as a full HTML page.

        Returns
        -------
        str :
            The HTML page.
        """
        return to_html(
            self._summary,
            standalone=True,
            column_filters=self.column_filters,
            **self._to_html_kwargs,
        )

    def html_snippet(self):
        """Get the report as an HTML fragment that can be inserted in a page.

        Returns
        -------
        str :
            The HTML snippet.
        """
        return to_html(
            self._summary,
            standalone=False,
            column_filters=self.column_filters,
            **self._to_html_kwargs,
        )

    def json(self):
        """Get the report data in JSON format.

        Returns
        -------
        str :
            The JSON data.
        """
        to_remove = ["dataframe", "sample_table"]
        data = {k: v for k, v in self._summary.items() if k not in to_remove}
        return json.dumps(data, cls=JSONEncoder)

    def _repr_mimebundle_(self, include=None, exclude=None):
        del include, exclude
        return {"text/html": self.html_snippet()}

    def _repr_html_(self):
        return self._repr_mimebundle_()["text/html"]

    def write_html(self, file):
        """Store the report into an HTML file.

        Parameters
        ----------
        file : str, pathlib.Path or file object
            The file object or path of the file to store the HTML output.
        """
        html = self.html()
        if isinstance(file, (str, Path)):
            with open(file, "w", encoding="utf8") as stream:
                stream.write(html)
            return

        try:
            # We don't have information about the write mode of the provided
            # file-object. We start by writing bytes into it.
            file.write(html.encode("utf-8"))
            return
        except TypeError:
            # We end-up here if the file-object was open in text mode
            # Let's give it another chance in this mode.
            pass

        if (encoding := getattr(file, "encoding", None)) is not None:
            try:
                encoding_name = codecs.lookup(encoding).name
            except LookupError:  # pragma: no cover
                encoding_name = None
            if encoding_name != "utf-8":
                raise ValueError(
                    "If `file` is a text file it should use utf-8 encoding; got:"
                    f" {encoding!r}"
                )
        # We write into the file-object expecting it to be in text mode at this
        # stage and with a UTF-8 encoding.
        file.write(html)

    def open(self):
        """Open the HTML report in a web browser."""
        open_in_browser(self.html())
