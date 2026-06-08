.. _sessionization:

.. |SessionEncoder| replace:: :class:`~skrub.SessionEncoder`
.. |BaseEstimator| replace:: :class:`~sklearn.base.BaseEstimator`
.. |TransformerMixin| replace:: :class:`~sklearn.base.TransformerMixin`


Detecting sessions in timestamped data with the SessionEncoder
----------------------------------------------------------------

When dealing with timestamped data (data that includes at least a timestamp column),
it may be beneficial to try and identify groups of events as
:ref:`"sessions" <https://en.wikipedia.org/wiki/Session_(web_analytics)>`_,
through **sessionization**.

Sessionization is the process of grouping a sequence of events (like user
interactions) into meaningful sessions.
For example, in an online retail context you might define a new session whenever
more than 30 minutes pass with no activity from the user. On a website, a session may
define a sequence of requests made by a single end-user within a certain time duration.

While definitions may vary depending on the specific use case, being able to detect
such "bursts" of activity by a user can often help with building features that have
greater predictive power than raw individual events, such as number of sessions or
average session duration.

The |SessionEncoder| addresses this problem by detecting sessions based on
a timestamp column, other session-related columns (e.g., user and device) that should be
used to distinguish between sessions, and a ``session_gap``. Session-related columns
-- identified by the ``split_by`` parameter -- allow to split sessions based on
the provided parameters, for example to group user actions only if they were conducted
on the same device.

A session is then defined as a sequence of events that share the same value in the
``split_by`` columns, and whose events are closer to each other than the
``session_gap``.

>>> from skrub import SessionEncoder
>>> from skrub.datasets import make_retail_events
>>> events = make_retail_events(n_events=100, random_state=0)
>>> X, y = events.X, events.y

Once the necessary features are provided, the |SessionEncoder|
returns a dataframe that includes a ``timestamp_session_id`` column, which is
composed of a monotonically increasing integer ID for each session:
>>> se = SessionEncoder(timestamp_col="timestamp", split_by="user_id", session_gap=30 * 60)
>>> res = se.fit_transform(X)
>>> res.head(5) # doctest: +SKIP
     user_id                        timestamp device_type page_category event_type  time_on_page  price_viewed  timestamp_session_id
0  user_0164 2024-01-01 03:29:07.708922+00:00      mobile       fashion  page_view         134.1        309.80                    59
1  user_0164 2024-01-01 03:29:42.185048+00:00      tablet         books     search         103.4         11.00                    59
2  user_0164 2024-01-01 03:32:38.352703+00:00     desktop          home   wishlist         180.3          4.80                    59
3  user_0008 2024-01-02 10:49:56.974375+00:00      mobile         books  page_view           7.0         33.94                     2
4  user_0149 2024-01-04 10:00:15.882835+00:00     desktop   electronics  page_view         108.5          4.44                    49

With the session ID, it becomes possible to compute aggregations on
each session, for example to find the duration of a session, or the number of sessions
by a user.

.. warning::

Aggregation can introduce data leakage! Records should only be aggregated from
within the training set at training time and the test set at predict time. To
ensure this is the case, any code that performs aggregation can be wrapped in a
scikit-learn |BaseEstimator| (as shown in the
:ref:`SessionEncoder example <sphx_glr_auto_examples_0110_session_encoder.py>`,
or the pipeline should use the skrub :ref:`Data Ops framework<user_guide_data_ops_plan>`.

The |SessionEncoder| includes the ``suffix`` parameter (by default
``suffix="session_id"``) to specify what the name of the new column should be.
This can help with creating multiple session IDs based on the same timestamp.
For example, we might want to create sessions based on users, and based on users
and their device:

>>> se = SessionEncoder(timestamp_col="timestamp",
... split_by="user_id",
... session_gap=30 * 60,
... suffix="user"
... )
>>> res = se.fit_transform(X)
>>> res.head(5) # doctest: +SKIP
     user_id                        timestamp  ... price_viewed timestamp_user
0  user_0164 2024-01-01 03:29:07.708922+00:00  ...       309.80             59
1  user_0164 2024-01-01 03:29:42.185048+00:00  ...        11.00             59
2  user_0164 2024-01-01 03:32:38.352703+00:00  ...         4.80             59
3  user_0008 2024-01-02 10:49:56.974375+00:00  ...        33.94              2
4  user_0149 2024-01-04 10:00:15.882835+00:00  ...         4.44             49

>>> se = SessionEncoder(timestamp_col="timestamp",
... split_by=["user_id", "device_type"],
... session_gap=30 * 60,
... suffix="user_device"
... )
>>> res = se.fit_transform(X)
>>> res.head(5) # doctest: +SKIP
     user_id                        timestamp  ... price_viewed timestamp_user_device
0  user_0164 2024-01-01 03:29:07.708922+00:00  ...       309.80                    75
1  user_0164 2024-01-01 03:29:42.185048+00:00  ...        11.00                    76
2  user_0164 2024-01-01 03:32:38.352703+00:00  ...         4.80                    74
3  user_0008 2024-01-02 10:49:56.974375+00:00  ...        33.94                     2
4  user_0149 2024-01-04 10:00:15.882835+00:00  ...         4.44                    59
