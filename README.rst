skrub
=======

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
--------------------

The goal of skrub is to facilitate building and deploying machine-learning
models on tabular data.

skrub provides high-level tools for joining dataframes (``Joiner``, ``AggJoiner``, ...),
encoding columns (``MinHashEncoder``, ``ToCategorical``, ...), building a pipeline
(``TableVectorizer``, ``tabular_learner``, ...), and more.

.. code-block:: shell

    >>> from skrub.datasets import fetch_employee_salaries
    >>> dataset = fetch_employee_salaries()
    >>> df = dataset.X
    >>> y = dataset.y
    # from a complex dataframe
    >>> df
       gender  department  department_name	division	assignment_category	employee_position_title	date_first_hired	year_first_hired
     0         F	         POL               Department of Police	MSB Information Mgmt and Tech Division Records...	Fulltime-Regular	Office Services Coordinator	09/22/1986	1986
     1         M	         POL	            Department of Police	ISB Major Crimes Division Fugitive Section	Fulltime-Regular	Master Police Officer	09/12/1988	1988
     ...       ...         	...	...	...	...	...	...	...
     9226      M	         CCL	            County Council	Council Central Staff	Fulltime-Regular	Manager II	09/05/2006	2006
     9227      M	         DLC	            Department of Liquor Control	Licensure, Regulation and Education	Fulltime-Regular	Alcohol/Tobacco Enforcement Specialist II	01/30/2012	2012

    # to a prediction
    >>> from sklearn.model_selection import cross_val_score
    >>> from skrub import tabular_learner
    >>> cross_val_score(tabular_learner('regressor'), df, y)

See our `examples <https://skrub-data.org/stable/auto_examples>`_.

What skrub cannot do
~~~~~~~~~~~~~~~~~~~~~~

`Semantic similarities <https://en.wikipedia.org/wiki/Semantic_similarity>`_
are currently not supported.
For example, the similarity between *car* and *automobile* is outside the reach
of the methods implemented here.

This kind of problem is tackled by
`Natural Language Processing <https://en.wikipedia.org/wiki/Natural_language_processing>`_
methods.

skrub can still help with handling typos and variations in this kind of setting.

Installation
------------

skrub can be easily installed via `pip` or conda. For more installation information, see
the `installation instructions <https://skrub-data.org/stable/install.html>`_.

Contributing
------------

The best way to support the development of skrub is to spread the word!

Also, if you already are a skrub user, we would love to hear about your use cases and challenges in the `Discussions <https://github.com/skrub-data/skrub/discussions>`_ section.

To report a bug or suggest enhancements, please
`open an issue <https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue>`_ and/or
`submit a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_.
