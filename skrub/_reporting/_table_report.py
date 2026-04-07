import codecs
import functools
import json
import numbers
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from skrub import selectors as s

from .. import _config
from .. import _dataframe as sbd
from ._html import to_html
from ._serve import open_in_browser
from ._summarize import summarize_dataframe
from ._utils import JSONEncoder


def _validate_plot_and_association(plot_distributions, compute_associations, n_columns):
    if plot_distributions is None:
        plot_distributions = "auto"
    if compute_associations is None:
        compute_associations = "auto"

    if plot_distributions not in (True, False, "auto"):
        raise ValueError(
            "'plot_distributions' must be True, False, or 'auto', got"
            f" {plot_distributions!r}."
        )

    if compute_associations not in (True, False, "auto"):
        raise ValueError(
            "'compute_associations' must be True, False, or 'auto', got"
            f" {compute_associations!r}."
        )

    if plot_distributions == "auto":
        plot_distributions = _config.get_config()["plots_threshold"] >= n_columns
    if compute_associations == "auto":
        compute_associations = (
            _config.get_config()["associations_threshold"] >= n_columns
        )

    return plot_distributions, compute_associations


def _check_col_filter(name, cols, df):
    err_msg = (
        "Custom column filters should be either a Selector or a list of column names"
        f"\n  or a list of column indices. Got a bad filter for key {name!r}: {cols!r}"
    )
    if isinstance(cols, s.Selector):
        return cols.expand_index(df)
    if not isinstance(cols, Sequence):
        raise TypeError(err_msg)
    if all(isinstance(c, str) for c in cols):
        all_col_names = set(sbd.column_names(df))
        bad_col_names = [c for c in cols if c not in all_col_names]
        if bad_col_names:
            raise ValueError(
                "The following column names passed for "
                f"filter {name!r} are not in the dataframe: {bad_col_names}"
            )
        return s.make_selector(cols).expand_index(df)
    if all(isinstance(c, numbers.Integral) for c in cols):
        bad_idx = [c for c in cols if not 0 <= c < sbd.shape(df)[1]]
        if bad_idx:
            raise ValueError(
                "The following column indices passed for "
                f"filter {name!r} are out of range: {bad_idx}"
            )
        return list(cols)
    raise TypeError(err_msg)


def _check_column_filters(column_filters, df):
    if column_filters is None:
        return None
    if not isinstance(column_filters, Mapping):
        raise TypeError(
            "column_filters should be a dict mapping names to column lists, "
            f"got object of type: {type(column_filters)}"
        )
    return {
        str(name): {
            "display_name": str(name),
            "columns": _check_col_filter(name, cols, df),
        }
        for name, cols in column_filters.items()
    }


class TableReport:
    r"""Summarize the contents of a dataframe.

    This class summarizes a dataframe or numpy array, providing information such as
    the type and summary statistics (mean, number of missing values, etc.) for each
    column. Numpy arrays are converted to pandas DataFrame or Series.

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
        Each key is the filter named to be displayed in the dropdown menu
        (e.g. ``"first_10"``), and the value is the desired filter. Allowed
        formats for the filter values are a list of column names,
        a list of column indices, or a Selector object.
        See the end of the "Examples" section below for details.
    verbose : int, default = 1
        Whether to print progress information while the report is being generated.

        * verbose = 1 prints how many columns have been processed so far.
        * verbose = 0 silences the output.
    plot_distributions : bool or "auto", default="auto"
        Whether to plot the distributions of the columns.

        - ``True``: always generate plots, regardless of column count.
        - ``False``: never generate plots.
        - ``"auto"`` (default): generate plots only when the number of columns
          does not exceed the configured ``plots_threshold``
          (see :func:`set_config`).

    compute_associations : bool or "auto", default="auto"
        Whether to compute associations between columns.

        - ``True``: always compute associations, regardless of column count.
        - ``False``: never compute associations.
        - ``"auto"`` (default): compute associations only when the number of
          columns does not exceed the configured ``associations_threshold``
          (see :func:`set_config`).

    open_tab : str, default="table"
        The tab that will be displayed by default when the report is opened.
        Must be one of "table", "stats", "distributions", or "associations".

        * "table": Shows a sample of the dataframe rows
        * "stats": Shows summary statistics for all columns
        * "distributions": Shows plots of column distributions
        * "associations": Shows column associations and similarities

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
    ...         "display_name": ["a", "b"],
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
        verbose=None,
        plot_distributions="auto",
        compute_associations="auto",
        open_tab="table",
    ):
        if isinstance(dataframe, np.ndarray):
            if dataframe.ndim == 1:
                dataframe = pd.Series(dataframe, name="0")

            elif dataframe.ndim == 2:
                dataframe = pd.DataFrame(
                    dataframe, columns=[str(i) for i in range(dataframe.shape[1])]
                )

            else:
                raise ValueError(
                    f"Input NumPy array has {dataframe.ndim} dimensions. "
                    "TableReport only supports 1D and 2D arrays"
                )

        n_rows = max(1, n_rows)
        if verbose is None:
            self.verbose = _config.get_config()["table_report_verbosity"]
        else:
            self.verbose = verbose

        # Validate open_tab parameter
        valid_tabs = ["table", "stats", "distributions", "associations"]
        if open_tab not in valid_tabs:
            raise ValueError(
                f"'open_tab' must be one of {valid_tabs}, got {open_tab!r}."
            )
        self.open_tab = open_tab

        self._summary_kwargs = {
            "order_by": order_by,
            "max_top_slice_size": -(n_rows // -2),
            "max_bottom_slice_size": n_rows // 2,
            "verbose": self.verbose,
        }
        self._to_html_kwargs = {}
        self.title = title
        self.column_filters = _check_column_filters(column_filters, dataframe)
        self.verbose = verbose
        self.dataframe = (
            sbd.to_frame(dataframe) if sbd.is_column(dataframe) else dataframe
        )
        self.n_columns = sbd.shape(self.dataframe)[1]
        (
            self.plot_distributions,
            self.compute_associations,
        ) = _validate_plot_and_association(
            plot_distributions,
            compute_associations,
            self.n_columns,
        )

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
        self.compute_associations = False
        self.plot_distributions = False
        # In minimal mode, fall back to 'table' if user selected unavailable tabs
        if self.open_tab in ["distributions", "associations"]:
            self.open_tab = "table"

    def _display_subsample_hint(self):
        self._summary["is_subsampled"] = True

    def __repr__(self):
        return f"<{self.__class__.__name__}: use .open() to display>"

    @functools.cached_property
    def _summary(self):
        return summarize_dataframe(
            self.dataframe,
            with_plots=self.plot_distributions,
            with_associations=self.compute_associations,
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
            open_tab=self.open_tab,
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
            open_tab=self.open_tab,
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
