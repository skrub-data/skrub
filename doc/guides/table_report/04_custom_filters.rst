.. |TableReport| replace:: :class:`~skrub.TableReport`


How to define custom filters for the TableReport
================================================

It is possible to define custom filters for the |TableReport| using either column
names, or :ref:`skrub selectors< user_guide_selectors>`.

By defining a custom filter, it becomes easier to show and work directly on a given
subset of columns.

For example, we might want to select only the columns whose name follows a certain
pattern (here, starting with "metric"):

>>> import pandas as pd
>>> from skrub import TableReport
>>> from skrub import selectors as s
>>> df = pd.DataFrame(
...     {"id": [1, 2, 3], "metric1": [1, 2, 3], "metric2": [4, 5, 6], "metric3": [7, 8, 9]}
... )

Custom filters should be defined as a dictionary where the key is the name of the
filter that should be displayed in the generated report, and the value is either
a list of columns, the index of the columns (first column has index 0 etc.), or
a skrub selector like in this case:

>>> filters = {"only_metrics": s.glob("metric*")}
>>> report = TableReport(df, column_filters=filters)

Custom filters are placed at the top of the list of filters, in the "Filter columns"
drop-down menu.
