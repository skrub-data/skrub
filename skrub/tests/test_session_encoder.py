import datetime
from functools import partial

import numpy as np
import pytest

from skrub import SessionEncoder
from skrub import _dataframe as sbd
from skrub._session_encoder import (
    _add_session_id,
    _factorize_column,
)


@pytest.fixture
def example_session_data(df_module):
    """Create example session data with multiple users and sessions."""
    timestamps = []
    user_ids = []
    usernames = []

    base_time = datetime.datetime(2024, 1, 1)

    # User 101, alice: 3 sessions with 5 events each, 10 days apart
    for session in range(3):
        session_start = base_time + datetime.timedelta(days=session * 10)
        for event in range(5):
            timestamps.append(session_start + datetime.timedelta(minutes=event * 2))
            user_ids.append(101)
            usernames.append("alice")

    # User 102, bob: 2 sessions with 3 events each, 2 hours apart
    for session in range(2):
        session_start = base_time + datetime.timedelta(days=35, hours=session * 2)
        for event in range(3):
            timestamps.append(session_start + datetime.timedelta(minutes=event * 5))
            user_ids.append(102)
            usernames.append("bob")

    # User 103, charlie: 1 session with 4 events
    session_start = base_time + datetime.timedelta(days=40)
    for event in range(4):
        timestamps.append(session_start + datetime.timedelta(minutes=event * 3))
        user_ids.append(103)
        usernames.append("charlie")

    return df_module.make_dataframe(
        {
            "timestamp": timestamps,
            "user_id": user_ids,
            "username": usernames,
        }
    )


@pytest.fixture
def example_session_data_multi_by(df_module):
    """Create example session data where a user is identified by two columns.

    A user is uniquely identified by the combination of ``user_id`` and
    ``device_id``.  The same ``user_id`` on two different devices produces
    independent sessions, which lets us verify that ``group_by`` accepts a list of
    column names.
    """
    timestamps = []
    user_ids = []
    device_ids = []

    base_time = datetime.datetime(2024, 1, 1)

    # user 1, device "mobile": 2 sessions, 10 days apart, 4 events each
    for session in range(2):
        session_start = base_time + datetime.timedelta(days=session * 10)
        for event in range(4):
            timestamps.append(session_start + datetime.timedelta(minutes=event * 3))
            user_ids.append(1)
            device_ids.append("mobile")

    # user 1, device "desktop": 1 session, 3 events
    # (same user_id as above but different device → separate sessions)
    session_start = base_time + datetime.timedelta(days=5)
    for event in range(3):
        timestamps.append(session_start + datetime.timedelta(minutes=event * 4))
        user_ids.append(1)
        device_ids.append("desktop")

    # user 2, device "mobile": 1 session, 5 events
    session_start = base_time + datetime.timedelta(days=20)
    for event in range(5):
        timestamps.append(session_start + datetime.timedelta(minutes=event * 2))
        user_ids.append(2)
        device_ids.append("mobile")

    return df_module.make_dataframe(
        {
            "timestamp": timestamps,
            "user_id": user_ids,
            "device_id": device_ids,
        }
    )


@pytest.mark.parametrize(
    "by_column,expected_sessions,group_key_to_sessions",
    [
        ("user_id", 6, {101: 3, 102: 2, 103: 1}),
        ("username", 6, {"alice": 3, "bob": 2, "charlie": 1}),
    ],
)
def test_session_encoder_basic(
    example_session_data, by_column, expected_sessions, group_key_to_sessions
):
    """Test basic sessionization grouping by user_id or username."""
    # Apply SessionEncoder grouping by the specified column
    se = SessionEncoder(group_by=by_column, timestamp_col="timestamp", session_gap=30)
    result = se.fit_transform(example_session_data)

    # Check that we have the expected total number of sessions
    session_ids = sbd.to_list(sbd.col(result, "timestamp_session_id"))
    unique_sessions = set(session_ids)
    assert len(unique_sessions) == expected_sessions

    # content of the "session_id" column after sessionization
    session_ids = sbd.to_list(sbd.col(result, "timestamp_session_id"))
    # content of the "by" column (user_id or username)
    group_values = sbd.to_list(sbd.col(result, by_column))

    counted_sessions = {}
    for group_key, session_id in zip(group_values, session_ids):
        if group_key not in counted_sessions:
            counted_sessions[group_key] = set()
        counted_sessions[group_key].add(session_id)
    for group_key, sessions in counted_sessions.items():
        assert len(sessions) == group_key_to_sessions[group_key]


@pytest.mark.parametrize(
    "by_column,group_keys",
    [
        ("user_id", [101, 102, 103]),
        ("username", ["alice", "bob", "charlie"]),
    ],
)
def test_session_encoder_different_users_different_sessions(
    example_session_data, by_column, group_keys
):
    """Test that different users/groups have different session IDs."""
    # Apply SessionEncoder
    se = SessionEncoder(group_by=by_column, timestamp_col="timestamp", session_gap=30)
    result = se.fit_transform(example_session_data)

    # content of the "session_id" column after sessionization
    session_ids = sbd.to_list(sbd.col(result, "timestamp_session_id"))
    # content of the "by" column (user_id or username)
    group_values = sbd.to_list(sbd.col(result, by_column))

    # Verify different groups don't share session IDs
    for i, key1 in enumerate(group_keys):
        for key2 in group_keys[i + 1 :]:
            # find the indices of events for each group key (user id or username)
            indices1 = [idx for idx, v in enumerate(group_values) if v == key1]
            indices2 = [idx for idx, v in enumerate(group_values) if v == key2]
            # find the unique session IDs for each group key (each user)
            sessions1 = {session_ids[idx] for idx in indices1}
            sessions2 = {session_ids[idx] for idx in indices2}

            # check that there are no shared session IDs between different users/groups
            assert len(sessions1.intersection(sessions2)) == 0


def test_session_encoder_multi_by_columns(example_session_data_multi_by):
    """Test sessionization when a user is identified by a combination of columns.

    The fixture has user_id=1 on two devices ("mobile" and "desktop").  When
    ``group_by=["user_id", "device_id"]``, those two device contexts must be treated
    as independent groups, producing separate session IDs even though they share
    the same ``user_id``.

    Expected sessions:
    - (user_id=1, device_id="mobile")  → 2 sessions
    - (user_id=1, device_id="desktop") → 1 session
    - (user_id=2, device_id="mobile")  → 1 session
    Total: 4 sessions
    """
    se = SessionEncoder(
        group_by=["user_id", "device_id"], timestamp_col="timestamp", session_gap=30
    )
    result = se.fit_transform(example_session_data_multi_by)

    session_ids = sbd.to_list(sbd.col(result, "timestamp_session_id"))
    user_ids = sbd.to_list(sbd.col(result, "user_id"))
    device_ids = sbd.to_list(sbd.col(result, "device_id"))

    # 4 distinct sessions overall
    assert len(set(session_ids)) == 4

    # create a dict that groups sessions by (user_id, device_id) pair
    group_sessions: dict = {}
    for uid, did, sid in zip(user_ids, device_ids, session_ids):
        key = (uid, did)
        # Each (user_id, device_id) pair should have its own set of session IDs
        # We use a set to track unique session IDs for each group key
        group_sessions.setdefault(key, set()).add(sid)

    # assert that each (user_id, device_id) pair has the expected number of sessions
    assert len(group_sessions[(1, "mobile")]) == 2
    assert len(group_sessions[(1, "desktop")]) == 1
    assert len(group_sessions[(2, "mobile")]) == 1

    # sessions belonging to different (user_id, device_id) pairs must be disjoint
    keys = list(group_sessions)
    # go through each pair of group keys (user_id, device_id)
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            # check that the sets in group_sessions for different keys are disjoint
            assert group_sessions[k1].isdisjoint(group_sessions[k2])


def test_session_encoder_multiple_users(df_module):
    """Test sessionization with multiple users interleaved."""
    timestamps = []
    user_ids = []

    base_time = datetime.datetime(2024, 1, 1)

    # Create events for two users, alternating
    for i in range(10):
        timestamps.append(base_time + datetime.timedelta(minutes=i))
        user_ids.append(101 if i % 2 == 0 else 102)

    df = df_module.make_dataframe(
        {
            "timestamp": timestamps,
            "user_id": user_ids,
        }
    )

    se = SessionEncoder(group_by="user_id", timestamp_col="timestamp", session_gap=30)
    result = se.fit_transform(df)

    # After sorting by user_id and timestamp, each user should have 1 session
    # since all their events are within 30 minutes
    session_ids = sbd.col(result, "timestamp_session_id")

    # The encoder sorts by user_id then timestamp, so events are grouped by user
    # Check that there are exactly 2 sessions (one per user)
    assert len(set(session_ids)) == 2


def test_session_encoder_time_gap_threshold(df_module):
    """Test that session_gap parameter correctly determines sessionization."""
    timestamps = [
        datetime.datetime(2024, 1, 1, 10, 0),
        datetime.datetime(2024, 1, 1, 10, 15),  # 15 min gap
        datetime.datetime(2024, 1, 1, 10, 50),  # 35 min gap
        datetime.datetime(2024, 1, 1, 11, 0),  # 10 min gap
    ]
    user_ids = [101, 101, 101, 101]

    df = df_module.make_dataframe(
        {
            "timestamp": timestamps,
            "user_id": user_ids,
        }
    )

    # With 20-minute gap: should create 2 sessions (split at 35-min gap)
    se_20 = SessionEncoder(
        group_by="user_id", timestamp_col="timestamp", session_gap=20
    )
    result_20 = se_20.fit_transform(df)
    session_ids_20 = sbd.to_list(sbd.col(result_20, "timestamp_session_id"))
    assert len(set(session_ids_20)) == 2

    # With 40-minute gap: should create 1 session (all gaps < 40 min)
    se_40 = SessionEncoder(
        group_by="user_id", timestamp_col="timestamp", session_gap=40
    )
    result_40 = se_40.fit_transform(df)
    session_ids_40 = sbd.to_list(sbd.col(result_40, "timestamp_session_id"))
    assert len(set(session_ids_40)) == 1


def test_session_encoder_no_user_column(df_module):
    """Test sessionization without a user identifier column.

    When ``group_by`` is None, all events are treated as from the same "user", and
    sessions are separated only by time gaps.
    """
    timestamps = [
        datetime.datetime(2024, 1, 1, 10, 0),
        datetime.datetime(2024, 1, 1, 10, 10),  # 10 min gap
        datetime.datetime(2024, 1, 1, 10, 15),  # 5 min gap (within 30 min)
        datetime.datetime(2024, 1, 1, 11, 0),  # 45 min gap (exceeds 30 min)
        datetime.datetime(2024, 1, 1, 11, 10),  # 10 min gap (within 30 min)
    ]

    df = df_module.make_dataframe(
        {
            "timestamp": timestamps,
        }
    )

    # Without 'group_by', sessions are separated only by time gaps
    se = SessionEncoder(group_by=None, timestamp_col="timestamp", session_gap=30)
    result = se.fit_transform(df)

    session_ids = sbd.to_list(sbd.col(result, "timestamp_session_id"))
    # Expected: 2 sessions (events 0-2 in session 0, event 3 starts new session)
    # Then event 4 continues session 1
    assert len(set(session_ids)) == 2
    assert (
        session_ids[0] == session_ids[1] == session_ids[2]
    )  # First 3 events in session 0
    assert session_ids[3] == session_ids[4]  # Last 2 events in session 1
    assert session_ids[0] != session_ids[3]  # Sessions are different


def test_session_encoder_single_event(df_module):
    """Test sessionization with single event per user."""
    df = df_module.make_dataframe(
        {
            "timestamp": [datetime.datetime(2024, 1, 1, 10, 0)],
            "user_id": [101],
        }
    )

    se = SessionEncoder(group_by="user_id", timestamp_col="timestamp", session_gap=30)
    result = se.fit_transform(df)

    session_ids = sbd.to_list(sbd.col(result, "timestamp_session_id"))
    assert len(session_ids) == 1
    # Single event should create one session
    assert session_ids[0] == 0


def test_session_encoder_empty_dataframe(df_module):
    """Test sessionization with empty dataframe."""
    df = df_module.make_dataframe(
        {
            "timestamp": [],
            "user_id": [],
        }
    )

    se = SessionEncoder(group_by="user_id", timestamp_col="timestamp", session_gap=30)
    result = se.fit_transform(df)

    assert sbd.shape(result)[0] == 0
    assert "timestamp_session_id" in sbd.column_names(result)


@pytest.mark.parametrize(
    "group_by_param,timestamp_col_param,expected_error_type,expected_error_match",
    [
        (
            "wrong_column",
            "timestamp",
            ValueError,
            "Column 'wrong_column' not found",
        ),
        (
            "user_id",
            "wrong_column",
            ValueError,
            "Column 'wrong_column' not found",
        ),
        (
            ["wrong_column", "user_device"],
            "timestamp",
            ValueError,
            "Column 'wrong_column' not found",
        ),
        (
            23,  # invalid type for 'group_by'
            "timestamp",
            TypeError,
            "group_by must be a string, a list of strings, or None",
        ),
    ],
)
def test_session_encoder_missing_column_error(
    df_module,
    group_by_param,
    timestamp_col_param,
    expected_error_type,
    expected_error_match,
):
    """Test that missing columns and invalid parameters raise appropriate errors."""
    df = df_module.make_dataframe(
        {
            "timestamp": [datetime.datetime(2024, 1, 1)],
            "user_id": [101],
            "user_device": ["mobile"],
        }
    )

    se = SessionEncoder(
        group_by=group_by_param,
        timestamp_col=timestamp_col_param,
    )
    with pytest.raises(expected_error_type, match=expected_error_match):
        se.fit_transform(df)


def test_session_encoder_invalid_parameters(df_module):
    """Test that invalid parameters raise appropriate errors."""
    df = df_module.make_dataframe(
        {
            "timestamp": [datetime.datetime(2024, 1, 1)],
            "user_id": [101],
        }
    )

    # Test negative session_gap
    se_negative = SessionEncoder(
        group_by="user_id", timestamp_col="timestamp", session_gap=-10
    )
    with pytest.raises(ValueError, match="session_gap must be a positive number"):
        se_negative.fit_transform(df)

    # Test zero session_gap
    se_zero = SessionEncoder(
        group_by="user_id", timestamp_col="timestamp", session_gap=0
    )
    with pytest.raises(ValueError, match="session_gap must be a positive number"):
        se_zero.fit_transform(df)

    # Test non-numeric session_gap
    se_non_numeric = SessionEncoder(
        group_by="user_id", timestamp_col="timestamp", session_gap="thirty"
    )
    with pytest.raises(ValueError, match="session_gap must be a positive number"):
        se_non_numeric.fit_transform(df)


def test_session_encoder_preserves_columns(df_module):
    """Test that original columns are preserved in output."""
    df = df_module.make_dataframe(
        {
            "timestamp": [
                datetime.datetime(2024, 1, 1, 10, 0),
                datetime.datetime(2024, 1, 1, 10, 5),
            ],
            "user_id": [101, 101],
            "extra_col": [1.5, 2.5],
        }
    )

    se = SessionEncoder(group_by="user_id", timestamp_col="timestamp", session_gap=30)
    result = se.fit_transform(df)

    result_cols = sbd.column_names(result)
    assert "timestamp" in result_cols
    assert "user_id" in result_cols
    assert "extra_col" in result_cols
    assert "timestamp_session_id" in result_cols


def test_session_encoder_fit_and_transform(df_module):
    """Test that fit() and transform() work separately."""
    df = df_module.make_dataframe(
        {
            "timestamp": [
                datetime.datetime(2024, 1, 1, 10, 0),
                datetime.datetime(2024, 1, 1, 10, 5),
            ],
            "user_id": [101, 101],
        }
    )

    se = SessionEncoder(group_by="user_id", timestamp_col="timestamp", session_gap=30)

    # Test fit returns self
    se_fitted = se.fit(df)
    assert se_fitted is se

    # Test that all_inputs_ is set after fit
    assert hasattr(se, "all_inputs_")


def test_get_feature_names(df_module):
    """Test that get_feature_names returns the correct list of columns."""
    df = df_module.make_dataframe(
        {
            "timestamp": [
                datetime.datetime(2024, 1, 1, 10, 0),
                datetime.datetime(2024, 1, 1, 10, 5),
            ],
            "user_id": [101, 101],
        }
    )

    se = SessionEncoder(group_by="user_id", timestamp_col="timestamp", session_gap=30)
    se.fit(df)
    feature_names = se.get_feature_names_out()

    # Should include original columns plus "timestamp_session_id"
    assert set(feature_names) == {"timestamp", "user_id", "timestamp_session_id"}


# ---------------------------------------------------------------------------
# Tests for the internal dispatched helper functions
# ---------------------------------------------------------------------------


def test_factorize_column_string(df_module):
    """_factorize_column should map string values to consecutive integer codes."""
    df = df_module.make_dataframe({"user": ["alice", "bob", "alice", "charlie"]})
    codes = _factorize_column(df, "user")

    # alice appears first, so it should get code 0
    assert codes[0] == codes[2]  # both "alice" → same code
    assert codes[1] != codes[0]  # "bob" differs from "alice"
    assert codes[3] != codes[0]  # "charlie" differs from "alice"
    assert codes[1] != codes[3]  # "bob" differs from "charlie"
    assert all(int(c) == expected for c, expected in zip(codes, [0, 1, 0, 2]))


def test_factorize_column_numeric(df_module):
    """_factorize_column on a numeric column should return the column unchanged."""
    df = df_module.make_dataframe({"user_id": [10, 20, 10, 30]})
    codes = _factorize_column(df, "user_id")

    df_module.assert_column_equal(codes, df["user_id"])


def test_check_is_new_session_no_by(df_module):
    """_check_is_new_session with an empty group_by-list uses only the time gap."""
    df = df_module.make_dataframe(
        {
            "timestamp": [
                datetime.datetime(2024, 1, 1, 10, 0),
                datetime.datetime(2024, 1, 1, 10, 10),  # 10 min — within gap
                datetime.datetime(2024, 1, 1, 11, 0),  # 50 min — exceeds gap
                datetime.datetime(2024, 1, 1, 11, 5),  # 5 min  — within gap
            ]
        }
    )
    session_id = sbd.to_list(
        sbd.col(_add_session_id(df, [], "timestamp", 30), "timestamp_session_id")
    )
    # Expected: first two events in session 0, last two events in session 1
    assert session_id == [0, 0, 1, 1]


def test_check_is_new_session_with_by(df_module):
    """_add_session_id returns a dataframe with a ``timestamp_session_id``
    column when a group_by-list is provided.  A new session starts when the group key
    changes (even for a tiny time gap) or when the time gap exceeds
    ``session_gap``.

    Data layout (already sorted by user_id, timestamp):
      row 0: user 1, 10:00 – first row, session 0
      row 1: user 1, 10:05 – same user, 5 min gap  → still session 0
      row 2: user 2, 10:06 – user changed, 1 min gap → new session 1
      row 3: user 2, 10:10 – same user, 4 min gap  → still session 1
    Expected session_ids: [0, 0, 1, 1]
    """
    df = df_module.make_dataframe(
        {
            "user_id": [1, 1, 2, 2],
            "timestamp": [
                datetime.datetime(2024, 1, 1, 10, 0),
                datetime.datetime(2024, 1, 1, 10, 5),  # same user, 5 min gap
                datetime.datetime(2024, 1, 1, 10, 6),  # different user, 1 min gap
                datetime.datetime(2024, 1, 1, 10, 10),  # same user, 4 min gap
            ],
        }
    )
    result = _add_session_id(df, ["user_id"], "timestamp", 30)

    # _add_session_id now returns the full dataframe with session_id added
    assert "timestamp_session_id" in sbd.column_names(result)
    session_ids = sbd.to_list(sbd.col(result, "timestamp_session_id"))
    assert session_ids == [0, 0, 1, 1]


@pytest.mark.parametrize(
    "func",
    (
        partial(
            _add_session_id, group_by=[], timestamp_col="timestamp", session_gap=30
        ),
        partial(_factorize_column, column_name="user_id"),
    ),
)
def test_error_dispatch(func):
    with pytest.raises(TypeError, match="Expecting a Pandas or Polars Dataframe"):
        func(np.array([1]))
