"""Summarize the contents of a dataframe and generate an HTML report."""

from ._patching import patch_display, unpatch_display
from ._table_report import TableReport

__all__ = ["TableReport", "patch_display", "unpatch_display"]
