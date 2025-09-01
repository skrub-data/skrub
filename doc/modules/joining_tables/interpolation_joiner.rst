Using the :class:`InterpolationJoiner` to join tables using ML predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :class:`InterpolationJoiner` is a transformer that performs an operation similar
to that of a regular equi-join, but that can handle the presence of missing rows
in the right table (the table to be added). This is done by estimating the value
that the missing rows would have by training a machine learning model on the data
we have access to.

This transformer is explored in more detail in :ref:`this example <sphx_glr_auto_examples_09_interpolation_join.py>`.
