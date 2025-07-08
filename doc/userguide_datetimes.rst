.. _userguide_datetimes
.. |ToDatetime| replace:: :class:`~skrub.ToDatetime`
.. |to_datetime| replace:: :func:`~skrub.to_datetime`
.. |DatetimeEncoder| replace:: :class:`~skrub.DatetimeEncoder`
========================================

Parsing and encoding datetimes
------------------

Parsing Datetime Strings with |ToDatetime| and |to_datetime|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``skrub`` includes objects to help with parsing and encoding datetimes.

- |to_datetime| and |ToDatetime| convert all columns in a dataframe that can be
parsed as datetimes to the proper dtype.
- |to_datetime| is a function; |ToDatetime| is a scikit-learn compatible transformer.

.. code-block:: python

    from skrub import to_datetime, ToDatetime
    s = pd.Series(["2024-05-05T13:17:52", None, "2024-05-07T13:17:52"], name="when")
    to_datetime(s)
    ToDatetime().fit_transform(s)

Encoding and Feature Engineering on Datetimes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once datetimes have been parsed, they can be encoded as numerical features with
the |DatetimeEncoder|. This encoder extracts temporal features (year, month, day,
hour, etc.) from datetime columns. No timezone conversion is done; the timezone
in the feature is retained. The |DatetimeEncoder| rejects non-datetime columns,
so it should only be applied after conversion using |ToDatetime|.

Besides extracting datetime features, |DatetimeEncoder| can include additional
time-based features, such as:

- Number of seconds from epoch (``add_total_seconds``)
- Day of the week (``add_weekday``)
- Day of the year (``add_day_of_year``)

Periodic encoding is supported through trigonometric (circular) and spline
encoding: set the ``periodic_encoding`` parameter to ``circular`` or ``spline``.
