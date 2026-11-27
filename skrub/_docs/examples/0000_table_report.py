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

.. image:: ../_static/table_report_demo.gif
    :alt: Animation showing clicks on the tabs, columns and cells of a
          TableReport

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
# .. admonition:: Dataframes, series and numpy arrays
#    :collapsible: closed
#
#    The |TableReport| accepts pandas or polars dataframes, but also series,
#    and 1-d or 2-d numpy arrays:
#
#    .. code-block:: python
#
#       TableReport(employees["gender"], open_tab="stats")
#       TableReport(employees["current_annual_salary"] / 12)
#       TableReport(employees[["year_first_hired", "current_annual_salary"]].to_numpy())
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
# The Stats tab
# ~~~~~~~~~~~~~
#
# The **Stats** tab shows summary statistics for each column, such as its
# dtype, its number of **Unique values** and **Null values**, or its mean.
#
# **Missing values** are highlighted in orange: for instance, ``gender``
# has |gender_nulls|.
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
# count above the plot. A value is shown as an outlier when it is far from most
# of the data, using a rule similar to the
# `interquartile range <https://en.wikipedia.org/wiki/Interquartile_range>`_:
# here, ``current_annual_salary`` has 2 outliers.

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
# .. image:: ../_static/table_report_filters_demo.gif
#    :alt: Animation showing the custom "Hiring date" filter in the "Filter
#          columns" menu of a TableReport

# %%
# Saving the report
# -----------------
#
# A report opened with ``.open()`` cannot be saved from the browser: use
# :meth:`~skrub.TableReport.write_html` to save it as an HTML file. The report
# can also be exported as JSON with :meth:`~skrub.TableReport.json`, as
# Markdown with :meth:`~skrub.TableReport.markdown`, or as a Python dictionary
# with :meth:`~skrub.TableReport.dict`. See
# :ref:`user_guide_table_report_sharing`.

# %%
# .. seealso::
#
#    :ref:`user_guide_table_report_customize`, to change the default behavior
#    of the reports with the skrub configuration.
