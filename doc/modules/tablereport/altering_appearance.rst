.. |TableReport| replace:: :class:`~skrub.TableReport`
.. |set_config| replace:: :func:`~skrub.set_config`
.. |column_associations| replace:: :func:`~skrub.column_associations`

Altering the Appearance of the |TableReport|
============================================

For performance reasons, the |TableReport| disables the computation of
distributions and associations for tables with more than 30 columns. This behavior
can be changed by modifying the ``max_plot_columns`` and ``max_association_columns``
parameter, or by altering the configuration with |set_config| (refer to the
|TableReport| and |set_config| docs for more detail).
