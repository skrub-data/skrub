.. |set_config| replace:: :func:`~skrub.set_config`
.. |config_context| replace:: :func:`~skrub.config_context`

.. _userguide_utils:

Customizing the behavior of skrub
=================================


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


The configuration parameters that can be set with ``set_config`` and ``config_context``
are available by using

>>> import skrub
>>> config = skrub.get_config()
>>> config.keys()
dict_keys(['use_table_report', 'use_table_report_data_ops', 'max_plot_columns', 'max_association_columns', 'subsampling_seed', 'enable_subsampling', 'float_precision', 'cardinality_threshold'])

These are the parameters currently available in the global configuration:
- **use_table_report**  (env variable: ``SKB_USE_TABLE_REPORT``)
  Default: ``False``. If set to ``True``, the html representation of Pandas and
  Polars dataframes is replaced with the :class:`~skrub.TableReport`.

- **use_table_report_data_ops** (env variable: ``SKB_USE_TABLE_REPORT_DATA_OPS``)
  Default: ``True``. Set the HTML representation used for the Data Ops previews.
  If ``True``, use the :class:`~skrub.TableReport`, otherwise use the default
  Pandas or Polars representation.

- **max_plot_columns** (env variable: ``SKB_MAX_PLOT_COLUMNS``)
  Default: 30. If a dataframe has more columns than the value set here, the
  :class:`~skrub.TableReport` will skip generating the plots.

- **max_association_columns**  (env variable: ``SKB_MAX_ASSOCIATION_COLUMNS``)
  Default: 30. If a dataframe has more columns than the value set here, the
  :class:`~skrub.TableReport` will skip computing the associations.

- **subsampling_seed**  (env variable: ``SKB_SUBSAMPLING_SEED``)
  Set the random seed of subsampling in :func:`skrub.DataOp.skb.subsample()`,
  when ``how="random"`` is passed.

- **enable_subsampling**  (env variable: ``SKB_ENABLE_SUBSAMPLING``)
  Default: ``"default"``. Control the activation of subsampling in
  :func:`skrub.DataOp.skb.subsample()`.
  If ``"default"``, the behavior of :func:`skrub.DataOp.skb.subsample()` is used.
  If ``"disable"``, subsampling is never used, so skb.subsample becomes a no-op.
  If ``"force"``, subsampling is used in all DataOps evaluation modes (eval(), fit_transform, etc.).

- **float_precision**  (env variable: ``SKB_FLOAT_PRECISION``)
  Default: 3. Control the number of significant digits shown when formatting floats.
  Applies overall precision rather than fixed decimal places.

- **cardinality_threshold**  (env variable: ``SKB_CARDINALITY_THRESHOLD``)
  Default: 40. Set the ``cardinality_threshold`` argument of :class:`~skrub.TableVectorizer`.
  Additionally, set the threshold for warning the user about high cardinality
  features in the :class:`~skrub.TableReport`.
