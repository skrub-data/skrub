.. _sessionization:

.. |SessionEncoder| replace:: :class:`~skrub.SessionEncoder`
.. |BaseEstimator| replace:: :class:`~sklearn.base.BaseEstimator`
.. |TransformerMixin| replace:: :class:`~sklearn.base.TransformerMixin`


Detecting sessions in timestamped data with the SessionEncoder
----------------------------------------------------------------

When dealing with timestamped data (data that includes at least a timestamp column),
it may be beneficial to try and identify groups of events through **sessionization**.

Sessionization is the process of grouping a sequence of events (like user
interactions) into meaningful sessions. A session typically starts fresh or
after a period of inactivity.

For example, in an online retail context, you might define a new session whenever
more than 30 minutes pass with no activity from a user. On a website, a session may
define a sequence of requests made by a single end-user within a certain time duration.

While definitions may vary depending on the specific use case, being able to detect
such "bursts" of activity by a user can help with building features that often have
greater predictive power than raw individual events.

The |SessionEncoder| helps addressing this problem by detecting sessions based on
a timestamp column, other "session columns" (e.g., user and device) that should be
used to distinguish between sessions, and a ``session_gap``. A session is then
defined as a sequence of events that share the same value in the "session columns"
and whose events are closer to each other than the ``session_gap``.

>>> from skrub import SessionEncoder
>>> from skrub.datasets import make_retail_events
>>> events = make_retail_events(n_events=100, random_state=0)
>>> X, y = events.X, events.y

Once the necessary features are provided, the |SessionEncoder|
returns a dataframe that includes a ``session_id`` column, which includes an integer,
monotonically increasing ID, for each session:

>>> se = SessionEncoder(timestamp_col="timestamp", split_by="user_id", session_gap=30 * 60)
>>> res = se.fit_transform(X)
>>> res.head(5)
     user_id                        timestamp device_type page_category event_type  time_on_page  price_viewed  timestamp_session_id
0  user_0164 2024-01-01 03:29:07.708922+00:00      mobile       fashion  page_view         134.1        309.80                    59
1  user_0164 2024-01-01 03:29:42.185048+00:00      tablet         books     search         103.4         11.00                    59
2  user_0164 2024-01-01 03:32:38.352703+00:00     desktop          home   wishlist         180.3          4.80                    59
3  user_0008 2024-01-02 10:49:56.974375+00:00      mobile         books  page_view           7.0         33.94                     2
4  user_0149 2024-01-04 10:00:15.882835+00:00     desktop   electronics  page_view         108.5          4.44                    49

Once the session ID is available, it becomes possible to compute aggregations on
each session, for example to find the duration of a session, or the number of sessions
by a user.

.. warning::

Aggregation can introduce data leakage! Records should only be aggregated from
within the training set at training time and the test set at predict time. To
ensure this is the case, any code that performs aggregation can be wrapped in a
scikit-learn |BaseEstimator| (as shown in the
:ref:`SessionEncoder example <sphx_glr_auto_examples_0110_session_encoder.py>`,
or the pipeline should use the skrub :ref:`Data Ops framework<user_guide_data_ops_plan>`.
