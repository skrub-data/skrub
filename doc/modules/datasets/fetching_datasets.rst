Fetching the example datasets used in ``skrub``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``skrub`` includes a number of datasets used for running examples. Each dataset
can be downloaded using its ``fetch_*`` function, provided in the ``skrub.datasets``
namespace:

.. code-block:: python

    from skrub.datasets import fetch_employee_salaries
    data = fetch_employee_salaries()

Datasets are stored as :class:`~sklearn.utils.Bunch` objects, which include the
full data, an ``X`` feature matrix, and a ``y`` target column with type ``pd.DataFrame``.
Some datasets may have a different format depending on the use case.
