
.. |set_config| replace:: :func:`~skrub.set_config`
.. |config_context| replace:: :func:`~skrub.config_context`

.. _userguide_utils:

Example datasets, utilities, and customization
-----------------------------------

Customizing the default configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skrub includes a configuration manager that allows setting various parameters (see the |set_config| documentation for more detail).

It is possible to change configuration options using the |set_config| function:

.. code-block:: python

    from skrub import set_config
    set_config(use_table_report=True)

Each configuration parameter can also be modified by setting its environment variable.

A |config_context| is also provided, which allows temporarily altering the configuration:

.. code-block:: python

    import skrub
    with skrub.config_context(max_plot_columns=1):
        ...

Fetching the example datasets used in ``skrub``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``skrub`` includes a number of datasets used for running examples. Each dataset can be downloaded using its ``fetch_*`` function, provided in the ``skrub.datasets`` namespace:

.. code-block:: python

    from skrub.datasets import fetch_employee_salaries
    data = fetch_employee_salaries()

Datasets are stored as :class:`~sklearn.utils.Bunch` objects, which include the full data, an ``X`` feature matrix, and a ``y`` target column with type ``pd.DataFrame``. Some datasets may have a different format depending on the use case.

Modifying the download location of ``skrub`` datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, datasets are stored in ``~/skrub_data``, where ``~`` is expanded as the (OS dependent) home directory of the user. The function ``get_data_dir`` shows the location that ``skrub`` uses to store data.

If needed, it is possible to change this location by modifying the environment variable ``SKRUB_DATA_DIRECTORY`` to an **absolute directory path**.
