Customizing the default configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skrub includes a configuration manager that allows setting various parameters
(see the |set_config| documentation for more detail).

It is possible to change configuration options using the |set_config| function:

>>> from skrub import set_config
>>> set_config(use_table_report=True)

Each configuration parameter can also be modified by setting its environment variable.

A |config_context| is also provided, which allows temporarily altering the configuration:

>>> import skrub
>>> with skrub.config_context(max_plot_columns=1):
...     pass
