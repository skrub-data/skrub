.. |TableReport| replace:: :class:`~skrub.TableReport`
.. |column_associations| replace:: :func:`~skrub.column_associations`

.. _user_guide_table_report_associations:

How to find correlated columns in a datarame
============================================

In addition to |TableReport|'s **Associations** tab, you can compute associations
using the |column_associations| function, which returns a dataframe containing the
associations.

Reported metrics include `Cramer’s V statistic <https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V>`_
and `Pearson’s Correlation Coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_.
The result is returned as a dataframe that contains the column name and idx for the
left and right table and both associations; results are sorted in descending order
by Cramer’s V association.

This can be useful to have access to the information used in the |TableReport|
for later use (e.g., to select which columns to drop).

.. code-block::

    from skrub import column_associations
    from skrub.datasets import fetch_employee_salaries
    import pandas as pd
    path = fetch_employee_salaries().path
    df = pd.read_csv(path)
    column_associations(df).head()

          left_column_name  left_column_idx        right_column_name  right_column_idx  cramer_v  pearson_corr
    0           department                1          department_name                 2  1.000000           NaN
    1  assignment_category                4    current_annual_salary                 8  0.635525           NaN
    2             division                3      assignment_category                 4  0.601097           NaN
    3  assignment_category                4  employee_position_title                 5  0.496814           NaN
    4             division                3  employee_position_title                 5  0.416034           NaN
