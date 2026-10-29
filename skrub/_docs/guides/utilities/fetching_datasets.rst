Working with the example datasets provided by skrub
-------------------------------------------------------

Skrub includes a number of datasets used for running examples. Each dataset
can be downloaded using its ``fetch_*`` function, provided in the ``skrub.datasets``
namespace:

.. code-block:: python

    from skrub.datasets import fetch_employee_salaries
    data = fetch_employee_salaries()

Datasets are stored as :class:`~sklearn.utils.Bunch` objects, which include a path
to each table in the dataset. Datasets should be loaded using the path:

.. code-block:: python

    import pandas as pd
    df = pd.read_csv(data.path)


Some datasets include multiple tables: in this case, ``path`` isn't available and
instead each table should be loaded with its own path:


.. code-block:: python

    from skrub.datasets import fetch_credit_fraud
    data = fetch_employee_salaries()
    baskets = pd.read_csv(data.baskets_path)
    products = pd.read_csv(data.products_path)


Modifying the download location of ``skrub`` datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, datasets are stored in ``~/skrub_data``, where ``~`` is expanded as
the (OS dependent) home directory of the user. The function
:func:`~skrub.datasets.get_data_dir` shows
the location that ``skrub`` uses to store data.

If needed, it is possible to change this location by modifying the environment
variable ``SKB_DATA_DIRECTORY`` to an **absolute directory path**.

See :ref:`user_guide_configuration_parameters` for more info on the global skrub
configuration.
