"""
Summarize the contents of a dataframe and generate an HTML report.
==================================================================

The ``TableReport`` provides an interactive HTML summary of a
dataframe (column statistics, histograms, correlations, missing values,
etc.).

The ``TableReport`` can be exported in HTML, JSON, or Markdown format.
"""

from ._patching import patch_display, unpatch_display
from ._table_report import TableReport

__all__ = ["TableReport", "patch_display", "unpatch_display"]
