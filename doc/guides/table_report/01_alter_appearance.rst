.. |TableReport| replace:: :class:`~skrub.TableReport`
.. |set_config| replace:: :func:`~skrub.set_config`
.. |column_associations| replace:: :func:`~skrub.column_associations`

.. _user_guide_table_report_customize:

How to tweak the Appearance of the |TableReport|
------------------------------------------------

The skrub global configuration includes various parameters that allow to tweak
the HTML representation of the |TableReport|.

For performance reasons, the |TableReport| disables the computation of
distributions and associations for tables with more than 30 columns.
This behavior can be overridden by setting the parameters ``plot_distributions``
and ``compute_associations`` to ``True`` respectively.

It is also possible to specify the floating point precision by setting the appropriate
``float_precision`` parameter.

The column threshold that is used by the |TableReport| can be modified in a given
script by using |set_config| and changing the values of
``table_report_plot_threshold`` and ``table_report_associations_threshold`` to
the desired threshold. Environment variables are also provided to set the threshold
permanently. Refer to :ref:`user_guide_configuration_parameters` for more detail.
