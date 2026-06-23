.. |set_config| replace:: :func:`~skrub.set_config`
.. |get_config| replace:: :func:`~skrub.get_config`
.. |config_context| replace:: :func:`~skrub.config_context`

.. _user_guide_configuration_parameters:

How to configure and customize the default behavior of skrub
============================================================


Skrub includes a configuration manager that allows setting various parameters
(see the |set_config| documentation for more detail).

It is possible to change configuration options using the |set_config| function:

>>> from skrub import set_config
>>> set_config(table_report_verbosity=0) # doctest: +SKIP

This alters the behavior of skrub in the current script. Each configuration parameter
has an environment variable that can be used to set it permanently.

Additionally, a |config_context| is provided to allow temporarily altering the
configuration:

>>> import skrub
>>> with skrub.config_context(table_report_plots_threshold=1):
...     pass

Within this context, only the code executed inside the ``with`` statement is affected.

The |get_config| function allows to retrieve the current configuration.

Configuration parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration parameters that can be set with ``set_config`` and ``config_context``
are available by using

>>> import skrub
>>> config = skrub.get_config()
>>> config.keys()
dict_keys(['use_table_report_data_ops', 'table_report_plots_threshold', 'table_report_associations_threshold', 'table_report_verbosity', 'subsampling_seed', 'enable_subsampling', 'float_precision', 'cardinality_threshold', 'data_dir', 'eager_data_ops', 'data_ops_open_graph_dropdown'])

These are the parameters currently available in the global configuration:

.. list-table:: Skrub Configuration Parameters
   :header-rows: 1
   :widths: 20 15 25 40

   * - Parameter Name
     - Default Value
     - Env Variable
     - Description
   * - ``use_table_report_data_ops``
     - ``True``
     - ``SKB_USE_TABLE_REPORT_DATA_OPS``
     - Set the HTML representation used for the Data Ops previews. If ``True``, use the :class:`~skrub.TableReport`, otherwise use the default Pandas or Polars representation.
   * - ``table_report_verbosity``
     - ``1``
     - ``SKB_TABLE_REPORT_VERBOSITY``
     - Set the verbosity of the :class:`~skrub.TableReport`. If ``1``, print on screen the progress by column, if ``0`` print nothing.
   * - ``table_report_plots_threshold``
     - 30
     - ``SKB_TABLE_REPORT_PLOTS_THRESHOLD``
     - If a dataframe has more columns than the value set here, the :class:`~skrub.TableReport` will skip generating the distribution plots (when ``plot_distributions="auto"``, the default).
   * - ``table_report_associations_threshold``
     - 30
     - ``SKB_TABLE_REPORT_ASSOCIATIONS_THRESHOLD``
     - If a dataframe has more columns than the value set here, the :class:`~skrub.TableReport` will skip computing the associations (when ``compute_associations="auto"``, the default).
   * - ``subsampling_seed``
     - 0
     - ``SKB_SUBSAMPLING_SEED``
     - Set the random seed of subsampling in :func:`skrub.DataOp.skb.subsample()`, when ``how="random"`` is passed.
   * - ``enable_subsampling``
     - ``"default"``
     - ``SKB_ENABLE_SUBSAMPLING``
     - Control the activation of subsampling in :func:`skrub.DataOp.skb.subsample()`. If ``"default"``, the behavior of :func:`skrub.DataOp.skb.subsample()` is used. If ``"disable"``, subsampling is never used, so skb.subsample becomes a no-op. If ``"force"``, subsampling is used in all DataOps evaluation modes (eval(), fit_transform, etc.).
   * - ``float_precision``
     - 3
     - ``SKB_FLOAT_PRECISION``
     - Control the number of significant digits shown when formatting floats. Applies overall precision rather than fixed decimal places.
   * - ``cardinality_threshold``
     - 40
     - ``SKB_CARDINALITY_THRESHOLD``
     - Set the ``cardinality_threshold`` argument of :class:`~skrub.TableVectorizer`. Additionally, set the threshold for warning the user about high cardinality features in the :class:`~skrub.TableReport`.
   * - ``data_dir``
     - ``~/skrub_data``
     - ``SKB_DATA_DIRECTORY``
     - Set the default location used by skrub to store datasets and other data, such as the Data Ops reports.
   * - ``eager_data_ops``
     - ``True``
     - ``SKB_EAGER_DATA_OPS``
     - Eagerly perform checks on the DataOps as soon they are created, and compute previews if preview data is available. If disabled, those checks are delayed until the DataOp is actually used
