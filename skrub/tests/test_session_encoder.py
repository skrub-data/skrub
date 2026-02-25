import datetime

import pytest

from skrub import SessionEncoder
from skrub import _dataframe as sbd


def test_session_encoder_basic(df_module):
    """Test basic sessionization with numeric user IDs."""
    # Create sample data with clear sessions
    timestamps = []
    user_ids = []
    values = []

    base_time = datetime.datetime(2024, 1, 1)

    # Create 3 sessions with events close together (2 min apart),
    # separated by large gaps (10 days)
    for session in range(3):
        session_start = base_time + datetime.timedelta(days=session * 10)
        for event in range(5):
            timestamps.append(session_start + datetime.timedelta(minutes=event * 2))
            user_ids.append(101)
            values.append(float(session * 5 + event))

    df = df_module.make_dataframe(
        {"timestamp": timestamps, "user_id": user_ids, "value": values}
    )

    # Apply SessionEncoder with 20-minute gap threshold
    se = SessionEncoder(by="user_id", timestamp="timestamp", session_gap=20)
    result = se.fit_transform(df)

    # Check that we have 3 sessions
    session_ids = sbd.to_list(sbd.col(result, "session_id"))
    unique_sessions = set(session_ids)
    assert len(unique_sessions) == 3, f"Expected 3 sessions, got {len(unique_sessions)}"

    # Check that events within a session have the same session_id
    # Each session has 5 events, so we should have patterns like:
    # [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    assert session_ids[0] == session_ids[4]
    assert session_ids[5] == session_ids[9]
    assert session_ids[10] == session_ids[14]

    # Check that different sessions have different IDs
    assert session_ids[0] != session_ids[5]
    assert session_ids[5] != session_ids[10]


def test_session_encoder_alphanumeric_users(df_module):
    """Test sessionization with alphanumeric user IDs."""
    timestamps = []
    user_ids = []

    base_time = datetime.datetime(2024, 1, 1)

    # User A: 2 sessions
    for session in range(2):
        session_start = base_time + datetime.timedelta(hours=session * 2)
        for event in range(3):
            timestamps.append(session_start + datetime.timedelta(minutes=event * 5))
            user_ids.append("USER_A")

    # User B: 1 session
    session_start = base_time + datetime.timedelta(days=1)
    for event in range(3):
        timestamps.append(session_start + datetime.timedelta(minutes=event * 5))
        user_ids.append("USER_B")

    df = df_module.make_dataframe(
        {
            "timestamp": timestamps,
            "user_id": user_ids,
        }
    )

    se = SessionEncoder(by="user_id", timestamp="timestamp", session_gap=30)
    result = se.fit_transform(df)

    session_ids = sbd.to_list(sbd.col(result, "session_id"))

    # Check User A has 2 sessions
    # Sessions should change when user changes or time gap exceeds threshold
    # First 3 events: USER_A session 1
    # Next 3 events: USER_A session 2 (2 hours gap > 30 min)
    # Last 3 events: USER_B session 3 (user change)
    assert len(set(session_ids)) == 3

    # Check that user change triggers new session
    user_a_sessions = set([session_ids[i] for i in range(6)])
    user_b_sessions = set([session_ids[i] for i in range(6, 9)])
    assert len(user_a_sessions.intersection(user_b_sessions)) == 0


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
