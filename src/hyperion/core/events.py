"""Event and Command envelope definitions for the event-driven architecture."""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class Envelope:
    """Base envelope for events and commands."""

    type: str
    data: dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))
    correlation_id: str | None = None
    causation_id: str | None = None
    aggregate_id: str | None = None  # Always experiment_id (for now)
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())


# Type aliases
Event = Envelope
Command = Envelope


class EventType:
    """Event type constants - facts that have occurred."""

    # Experiment lifecycle
    EXPERIMENT_STARTED = "EXPERIMENT_STARTED"
    EXPERIMENT_COMPLETED = "EXPERIMENT_COMPLETED"
    EXPERIMENT_PAUSED = "EXPERIMENT_PAUSED"
    EXPERIMENT_RESUMED = "EXPERIMENT_RESUMED"

    # Trial lifecycle
    TRIAL_STARTED = "TRIAL_STARTED"
    TRIAL_PROGRESS = "TRIAL_PROGRESS"
    TRIAL_COMPLETED = "TRIAL_COMPLETED"
    TRIAL_FAILED = "TRIAL_FAILED"
    TRIAL_KILLED = "TRIAL_KILLED"

    # System events
    DECISION_RECORDED = "DECISION_RECORDED"
    CAPACITY_AVAILABLE = "CAPACITY_AVAILABLE"


class CommandType:
    """Command type constants - intentions to change state."""

    # Experiment control
    START_EXPERIMENT = "START_EXPERIMENT"
    PAUSE_EXPERIMENT = "PAUSE_EXPERIMENT"
    RESUME_EXPERIMENT = "RESUME_EXPERIMENT"

    # Trial control
    START_TRIAL = "START_TRIAL"
    KILL_TRIAL = "KILL_TRIAL"
    PATCH_TRIAL = "PATCH_TRIAL"
