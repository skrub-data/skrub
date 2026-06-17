"""
Summarize the contents of a dataframe and generate an HTML report.
==================================================================

The :class:`~skrub.TableReport` provides an interactive HTML summary of a
dataframe (column statistics, histograms, correlations, missing values,
etc.).  :func:`~skrub.patch_display` can be used to replace the default
pandas / polars HTML repr with ``TableReport`` for a richer exploration
experience.

The public API is re-exported from the top-level ``skrub`` package:

- :class:`~skrub.TableReport`
- :func:`~skrub.patch_display`
- :func:`~skrub.unpatch_display`

Anything not listed in ``__all__`` is private and should not be used
directly.
"""

from ._patching import patch_display, unpatch_display
from ._table_report import TableReport

__all__ = ["TableReport", "patch_display", "unpatch_display"]
