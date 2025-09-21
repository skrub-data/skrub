Working with the example datasets provided by skrub
-------------------------------------------------------

skrub includes a number of datasets used for running examples. Each dataset
can be downloaded using its ``fetch_*`` function, provided in the ``skrub.datasets``
namespace:

.. code-block:: python

    from skrub.datasets import fetch_employee_salaries
    data = fetch_employee_salaries()

Datasets are stored as :class:`~sklearn.utils.Bunch` objects, which include the
full data, an ``X`` feature matrix, and a ``y`` target column with type ``pd.DataFrame``.
Some datasets may have a different format depending on the use case.

Modifying the download location of ``skrub`` datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, datasets are stored in ``~/skrub_data``, where ``~`` is expanded as
the (OS dependent) home directory of the user. The function ``get_data_dir`` shows
the location that ``skrub`` uses to store data.

If needed, it is possible to change this location by modifying the environment
variable ``SKRUB_DATA_DIRECTORY`` to an **absolute directory path**.

See :ref:`user_guide_configuration_parameters` for more info on the global skrub
configuration.
