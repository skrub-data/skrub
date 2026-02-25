import datetime

import pytest

from skrub import SessionEncoder
from skrub import _dataframe as sbd


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
    se = SessionEncoder(by=by_column, timestamp="timestamp", session_gap=30)
    result = se.fit_transform(example_session_data)

    # Check that we have the expected total number of sessions
    session_ids = sbd.to_list(sbd.col(result, "session_id"))
    unique_sessions = set(session_ids)
    assert len(unique_sessions) == expected_sessions

    # Get the appropriate column data based on what we're grouping by
    if by_column == "user_id":
        group_values = sbd.to_list(sbd.col(result, "user_id"))
    else:  # by_column == "username"
        group_values = sbd.to_list(sbd.col(result, "username"))

    counted_sessions = {}
    for group_key, session_id in zip(group_values, session_ids):
        if group_key not in counted_sessions:
            counted_sessions[group_key] = set()
        counted_sessions[group_key].add(session_id)
    for group_key, sessions in counted_sessions.items():
        assert len(sessions) == group_key_to_sessions[group_key]


@pytest.mark.parametrize(
    "by_column",
    ["user_id", "username"],
)
def test_session_encoder_different_users_different_sessions(
    example_session_data, by_column
):
    """Test that different users/groups have different session IDs."""
    # Apply SessionEncoder
    se = SessionEncoder(by=by_column, timestamp="timestamp", session_gap=30)
    result = se.fit_transform(example_session_data)

    session_ids = sbd.to_list(sbd.col(result, "session_id"))
    result_user_ids = sbd.to_list(sbd.col(result, "user_id"))
    result_usernames = sbd.to_list(sbd.col(result, "username"))

    # Get the appropriate column data based on what we're grouping by
    if by_column == "user_id":
        group_values = result_user_ids
        group_keys = [101, 102, 103]
    else:  # by_column == "username"
        group_values = result_usernames
        group_keys = ["alice", "bob", "charlie"]

    # Verify different groups don't share session IDs
    for i, key1 in enumerate(group_keys):
        for key2 in group_keys[i + 1 :]:
            indices1 = [idx for idx, v in enumerate(group_values) if v == key1]
            indices2 = [idx for idx, v in enumerate(group_values) if v == key2]
            sessions1 = set([session_ids[idx] for idx in indices1])
            sessions2 = set([session_ids[idx] for idx in indices2])
            assert len(sessions1.intersection(sessions2)) == 0, (
                f"Groups {key1} and {key2} should not share session IDs"
            )


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

    se = SessionEncoder(by="user_id", timestamp="timestamp", session_gap=30)
    result = se.fit_transform(df)

    # After sorting by user_id and timestamp, each user should have 1 session
    # since all their events are within 30 minutes
    session_ids = sbd.to_list(sbd.col(result, "session_id"))

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
    se_20 = SessionEncoder(by="user_id", timestamp="timestamp", session_gap=20)
    result_20 = se_20.fit_transform(df)
    session_ids_20 = sbd.to_list(sbd.col(result_20, "session_id"))
    assert len(set(session_ids_20)) == 2

    # With 40-minute gap: should create 1 session (all gaps < 40 min)
    se_40 = SessionEncoder(by="user_id", timestamp="timestamp", session_gap=40)
    result_40 = se_40.fit_transform(df)
    session_ids_40 = sbd.to_list(sbd.col(result_40, "session_id"))
    assert len(set(session_ids_40)) == 1


def test_session_encoder_single_event(df_module):
    """Test sessionization with single event per user."""
    df = df_module.make_dataframe(
        {
            "timestamp": [datetime.datetime(2024, 1, 1, 10, 0)],
            "user_id": [101],
        }
    )

    se = SessionEncoder(by="user_id", timestamp="timestamp", session_gap=30)
    result = se.fit_transform(df)

    session_ids = sbd.to_list(sbd.col(result, "session_id"))
    assert len(session_ids) == 1
    # Single event should create one session
    assert session_ids[0] in [0, 1]  # Could be 0 or 1 depending on implementation


def test_session_encoder_empty_dataframe(df_module):
    """Test sessionization with empty dataframe."""
    df = df_module.make_dataframe(
        {
            "timestamp": [],
            "user_id": [],
        }
    )

    se = SessionEncoder(by="user_id", timestamp="timestamp", session_gap=30)
    result = se.fit_transform(df)

    assert sbd.shape(result)[0] == 0
    assert "session_id" in sbd.column_names(result)


def test_session_encoder_missing_column_error(df_module):
    """Test that missing columns raise appropriate errors."""
    df = df_module.make_dataframe(
        {
            "timestamp": [datetime.datetime(2024, 1, 1)],
            "wrong_column": [101],
        }
    )

    se = SessionEncoder(by="user_id", timestamp="timestamp", session_gap=30)
    with pytest.raises(ValueError, match="Column 'user_id' not found"):
        se.fit_transform(df)

    df2 = df_module.make_dataframe(
        {
            "wrong_timestamp": [datetime.datetime(2024, 1, 1)],
            "user_id": [101],
        }
    )

    se2 = SessionEncoder(by="user_id", timestamp="timestamp", session_gap=30)
    with pytest.raises(ValueError, match="Column 'timestamp' not found"):
        se2.fit_transform(df2)


def test_session_encoder_invalid_parameters(df_module):
    """Test that invalid parameters raise appropriate errors."""
    df = df_module.make_dataframe(
        {
            "timestamp": [datetime.datetime(2024, 1, 1)],
            "user_id": [101],
        }
    )

    # Test negative session_gap
    se_negative = SessionEncoder(by="user_id", timestamp="timestamp", session_gap=-10)
    with pytest.raises(ValueError, match="session_gap must be a positive number"):
        se_negative.fit_transform(df)

    # Test zero session_gap
    se_zero = SessionEncoder(by="user_id", timestamp="timestamp", session_gap=0)
    with pytest.raises(ValueError, match="session_gap must be a positive number"):
        se_zero.fit_transform(df)


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

    se = SessionEncoder(by="user_id", timestamp="timestamp", session_gap=30)
    result = se.fit_transform(df)

    result_cols = sbd.column_names(result)
    assert "timestamp" in result_cols
    assert "user_id" in result_cols
    assert "extra_col" in result_cols
    assert "session_id" in result_cols


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

    se = SessionEncoder(by="user_id", timestamp="timestamp", session_gap=30)

    # Test fit returns self
    se_fitted = se.fit(df)
    assert se_fitted is se

    # Test that all_inputs_ is set after fit
    assert hasattr(se, "all_inputs_")
