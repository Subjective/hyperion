"""Policy interface and action definitions."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from hyperion.core.events import Event
from hyperion.core.state import ReadableState


@dataclass
class StartTrial:
    """Action to start a new trial."""

    experiment_id: str
    params: dict[str, Any]
    parent_trial_id: str | None = None
    tags: dict[str, Any] | None = None


@dataclass
class KillTrial:
    """Action to kill a running trial."""

    trial_id: str
    reason: str


@dataclass
class PatchTrial:
    """Action to patch trial parameters at runtime."""

    trial_id: str
    patch: dict[str, Any]


# Union type for all actions
Action = StartTrial | KillTrial | PatchTrial


class Policy(Protocol):
    """Protocol for optimization policies (strategies and agents).

    Policies receive events and produce actions based on system state.
    """

    experiment_id: str | None

    async def on_events(self, events: Iterable[Event]) -> None:
        """Process incoming events to update internal state.

        Args:
            events: Stream of events that have occurred
        """
        ...

    async def decide(self, state: ReadableState) -> list[Action]:
        """Decide what actions to take based on current state.

        Args:
            state: Read-only view of current system state

        Returns:
            List of actions to execute
        """
        ...

    async def rationale(self) -> str | None:
        """Get explanation for recent decisions.

        Returns:
            Human-readable rationale or None
        """
        ...
