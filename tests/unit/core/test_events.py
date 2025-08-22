"""Tests for Event and Command envelopes."""

from datetime import UTC, datetime

import pytest

from hyperion.core.events import Command, CommandType, Event, EventType


def test_event_is_immutable():
    """Event envelopes should be immutable after creation."""
    event = Event(type=EventType.TRIAL_STARTED, data={"trial_id": "123"})
    with pytest.raises(AttributeError):
        event.type = EventType.TRIAL_COMPLETED  # type: ignore[misc]


def test_event_generates_unique_ids():
    """Each event should have a unique ID."""
    e1 = Event(type=EventType.TRIAL_STARTED, data={})
    e2 = Event(type=EventType.TRIAL_STARTED, data={})
    assert e1.id != e2.id


def test_event_timestamp_defaults_to_now():
    """Events should default to current UTC time."""
    before = datetime.now(UTC)
    event = Event(type=EventType.TRIAL_STARTED, data={})
    after = datetime.now(UTC)

    assert before <= event.ts <= after


def test_correlation_causation_ids():
    """Events can track correlation and causation chains."""
    event = Event(
        type=EventType.TRIAL_COMPLETED,
        data={"score": 0.95},
        correlation_id="corr-123",
        causation_id="cmd-456",
    )
    assert event.correlation_id == "corr-123"
    assert event.causation_id == "cmd-456"


def test_command_envelope_same_as_event():
    """Commands use the same envelope structure as events."""
    cmd = Command(type=CommandType.START_TRIAL, data={"params": {"lr": 0.01}})
    assert hasattr(cmd, "id")
    assert hasattr(cmd, "ts")
    assert cmd.type == CommandType.START_TRIAL


def test_aggregate_id_for_entity_grouping():
    """Events can be grouped by aggregate_id (e.g., trial_id, experiment_id)."""
    event = Event(
        type=EventType.TRIAL_PROGRESS,
        data={"metrics": {"loss": 0.5}},
        aggregate_id="trial-789",
    )
    assert event.aggregate_id == "trial-789"


def test_event_metadata_field():
    """Events support arbitrary metadata."""
    event = Event(
        type=EventType.DECISION_RECORDED,
        data={"actions": []},
        metadata={"actor": "RandomSearchPolicy", "rationale": "exploration"},
    )
    assert event.metadata["actor"] == "RandomSearchPolicy"
