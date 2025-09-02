.. |TableReport| replace:: :class:`~skrub.TableReport`
.. |set_config| replace:: :func:`~skrub.set_config`

Altering the Appearance of the |TableReport|
============================================

The skrub global configuration includes various parameters that allow to tweak
the HTML representation of the |TableReport|.

For performance reasons, the |TableReport| disables the computation of
distributions and associations for tables with more than 30 columns. This behavior
can be changed by modifying the ``max_plot_columns`` and ``max_association_columns``
parameter.

It is also possible to specify the floating point precision by setting the appropriate
``float_precision`` parameter.

Parameters can be made permanent in a script by altering the configuration with
|set_config|, or by setting the respective environment variables. Refer to
:ref:`skrub_configuration_parameters` for more detail.
