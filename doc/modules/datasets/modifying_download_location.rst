Modifying the download location of ``skrub`` datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, datasets are stored in ``~/skrub_data``, where ``~`` is expanded as
the (OS dependent) home directory of the user. The function ``get_data_dir`` shows
the location that ``skrub`` uses to store data.

If needed, it is possible to change this location by modifying the environment
variable ``SKRUB_DATA_DIRECTORY`` to an **absolute directory path**.
