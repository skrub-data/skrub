.. _userguide_datetimes
.. |ToDatetime| replace:: :class:`~skrub.ToDatetime`
.. |to_datetime| replace:: :func:`~skrub.to_datetime`
.. |DatetimeEncoder| replace:: :class:`~skrub.DatetimeEncoder`
========================================

Parsing and encoding datetimes
------------------

Parsing Datetime Strings with |ToDatetime| and |to_datetime|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skrub provides helpers to parse datetime string columns automatically:

- The |to_datetime| function converts all columns in a dataframe that can be parsed as datetimes. The format can be inferred or user-specified with the `format` argument.
- The |ToDatetime| transformer follows the same logic during training and learns a mapping between columns and their formats. It then applies this mapping during the transform step.

.. code-block:: python

    from skrub import to_datetime, ToDatetime
    s = pd.Series(["2024-05-05T13:17:52", None, "2024-05-07T13:17:52"], name="when")
    to_datetime(s)
    ToDatetime().fit_transform(s)

Encoding and Feature Engineering on Datetimes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once datetime columns have been parsed, they can be encoded as numerical features with
the |DatetimeEncoder|, by extracting temporal features (year, month, day,
hour, etc.). No timezone conversion is done; the timezone
in the feature is retained. The |DatetimeEncoder| rejects non-datetime columns,
so it should only be applied after conversion using |ToDatetime|.

Additionally, |DatetimeEncoder| can include the following features:

- Number of seconds from epoch (``add_total_seconds``)
- Day of the week (``add_weekday``)
- Day of the year (``add_day_of_year``)

Periodic encoding is supported through trigonometric (circular) and spline
encoding: set the ``periodic_encoding`` parameter to ``circular`` or ``spline``.

.. figure:: /_static/periodic_features.png
    :alt: Periodic encoding of datetime features
    :align: center
    :width: 70%

    Example of periodic encoding of datetime features using circular and spline methods.
