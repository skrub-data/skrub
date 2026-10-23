"""
Exploring a dataframe with the TableReport
==========================================

The |TableReport| gives an interactive overview of a dataframe: a preview of
its rows, summary statistics for each column, plots of their distributions,
and the associations between columns. In this example, we go through its
features on the employee salaries dataset.

.. admonition:: Click on the reports!
    :class: important

    The reports displayed in this page are **interactive**. Most of their features only appear when you interact with them:
    as you read this example, click on each |TableReport| to try them out!

.. seealso::

    The :ref:`user guide <user_guide_table_report_start>` for a description of
    the |TableReport|.

.. |TableReport| replace:: :class:`~skrub.TableReport`
"""

# %%
# Overview of the report
# ----------------------
#
# Creating a |TableReport| only requires passing it the dataframe. In a script,
# ``TableReport(df).open()`` opens it in a browser.

import pandas as pd

from skrub import TableReport
from skrub.datasets import fetch_employee_salaries

employees = pd.read_csv(fetch_employee_salaries().path)
TableReport(employees)

# %%
# The report is organized in 4 tabs: **Table**, **Stats**, **Distributions**
# and **Associations**.
#
# .. admonition:: Dataframes and series
#    :collapsible: closed
#
#    The |TableReport| accepts pandas or polars dataframes, but also series:
#
#    .. code-block:: python
#
#       TableReport(employees["gender"], open_tab="stats")
#       TableReport(employees["current_annual_salary"] / 12)
#
# In the rest of this example, we look at each feature in detail.
#
# The Table tab
# ~~~~~~~~~~~~~
#
# The **Table** tab shows the first and last 5 rows of the dataframe, followed
# by its shape (number of rows and columns). The ``n_rows`` parameter changes
# the number of rows: with ``n_rows=4``, the report shows the first and last 2
# rows.

TableReport(employees, n_rows=4)

# %%
# Clicking on a column to see its details
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Clicking on a cell opens a summary of its column below the table. The arrow
# keys move the selection, and :kbd:`Esc` closes it.
#
# The copy button above the table (or :kbd:`Ctrl-C`) copies the Python
# representation of the value, e.g. ``'Department of Police'`` with quotes,
# ready to be pasted into your code.

# %%
# The Stats tab
# ~~~~~~~~~~~~~
#
# The **Stats** tab shows summary statistics for each column, such as its
# dtype, its number of **Unique values** and **Null values**, or its mean.
#
# **Missing values**: a single column can also be passed to the |TableReport|.
# For instance, ``gender`` has |gender_nulls|, shown in orange.
#
# .. |gender_nulls| raw:: html
#
#    <span style="color: #ff6600">17 null values</span>
#
# The buttons in the headers sort the table: sorting by **Unique values** shows
# that ``gender`` and ``assignment_category`` only have 2 distinct values.
# Here, ``open_tab`` opens the report directly on this tab:

TableReport(employees, n_rows=4, open_tab="stats")

# %%
# The Distributions tab
# ~~~~~~~~~~~~~~~~~~~~~
#
# The **Distributions** tab shows a card for each column, with a plot of its
# distribution and its main statistics. The checkboxes build a list of column
# names to copy into your code.
#
# There are two types of cards: string columns show their most frequent
# values, and numeric columns show a histogram.
#
# **Outliers** are grouped in a red bar at the edge of the histogram, with their
# count above the plot. To focus on ``current_annual_salary``, which has 2 of
# them, we can pass this single column to the |TableReport|:

TableReport(
    employees["current_annual_salary"], n_rows=4, open_tab="distributions"
)

# %%
# The Associations tab
# ~~~~~~~~~~~~~~~~~~~~
#
# The **Associations** tab lists the most associated pairs of columns, measured
# by Cramér's V and, for numeric columns, Pearson's correlation.

TableReport(employees, n_rows=4, open_tab="associations")

# %%
# Filtering the columns
# ---------------------
#
# The **Filter columns** menu restricts the columns shown in the **Table**,
# **Stats** and **Distributions** tabs. For instance, **With nulls** only keeps
# ``gender``.
#
# Your **own filters** can be added with ``column_filters``, a dictionary
# mapping filter names to lists of columns:

report = TableReport(
    employees,
    n_rows=4,
    column_filters={"Hiring date": ["date_first_hired", "year_first_hired"]},
)

# %%
# Configuring and saving the report
# ---------------------------------
#
# Configuring the report
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Options set with :func:`~skrub.set_config` apply to all the reports: for
# instance, the number of significant digits shown for numbers, or the number
# of columns above which plots and associations are skipped to keep the report
# fast. See :ref:`user_guide_table_report_customize`.
#
# Saving the report
# ~~~~~~~~~~~~~~~~~
#
# A report opened with ``.open()`` cannot be saved from the browser: use
# :meth:`~skrub.TableReport.write_html` to save it as an HTML file. The report
# can also be exported as JSON or Markdown. See
# :ref:`user_guide_table_report_sharing`.
