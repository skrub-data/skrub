skrub
=====

.. image:: https://skrub-data.github.io/stable/_static/skrub.svg
   :align: center
   :width: 50 %
   :alt: skrub logo


|py_ver| |pypi_var| |pypi_dl| |codecov| |circleci| |black|

.. |py_ver| image:: https://img.shields.io/pypi/pyversions/skrub
.. |pypi_var| image:: https://img.shields.io/pypi/v/skrub?color=informational
.. |pypi_dl| image:: https://img.shields.io/pypi/dm/skrub
.. |codecov| image:: https://img.shields.io/codecov/c/github/skrub-data/skrub/main
.. |circleci| image:: https://img.shields.io/circleci/build/github/skrub-data/skrub/main?label=CircleCI
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg


**skrub** (formerly *dirty_cat*) is a Python
library that facilitates prepping your tables for machine learning.

If you like the package, spread the word and ⭐ this repository!
You can also join the `discord server <https://discord.gg/ABaPnm7fDC>`_.

Website: https://skrub-data.org/

What can skrub do?
------------------

The goal of skrub is to bridge the gap between tabular data sources and machine-learning models.

skrub provides high-level tools for joining dataframes (``Joiner``, ``AggJoiner``, ...),
encoding columns (``MinHashEncoder``, ``ToCategorical``, ...), building a pipeline
(``TableVectorizer``, ``tabular_learner``, ...), and more.

>>> from skrub.datasets import fetch_employee_salaries
>>> dataset = fetch_employee_salaries()
>>> df = dataset.X
>>> y = dataset.y
>>> df.iloc[0]
gender                                                                     F
department                                                               POL
department_name                                         Department of Police
division                   MSB Information Mgmt and Tech Division Records...
assignment_category                                         Fulltime-Regular
employee_position_title                          Office Services Coordinator
date_first_hired                                                  09/22/1986
year_first_hired                                                        1986

>>> from sklearn.model_selection import cross_val_score
>>> from skrub import tabular_learner
>>> cross_val_score(tabular_learner('regressor'), df, y)
array([0.89370447, 0.89279068, 0.92282557, 0.92319094, 0.92162666])

See our `examples <https://skrub-data.org/stable/auto_examples>`_.

Installation
------------

skrub can easily be installed via ``pip`` or ``conda``. For more installation information, see
the `installation instructions <https://skrub-data.org/stable/install.html>`_.

Contributing
------------

The best way to support the development of skrub is to spread the word!

Also, if you already are a skrub user, we would love to hear about your use cases and challenges in the `Discussions <https://github.com/skrub-data/skrub/discussions>`_ section.

To report a bug or suggest enhancements, please
`open an issue <https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue>`_ and/or
`submit a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_.
