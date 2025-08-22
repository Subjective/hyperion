"""State interfaces and protocols for querying system state."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from hyperion.core.events import Event
from hyperion.core.models import Decision, Experiment, Trial


@dataclass(frozen=True)
class TrialView:
    """Immutable view of a trial's state."""

    trial_id: str
    experiment_id: str
    params: dict[str, Any]
    status: str
    score: float | None
    metrics_last: dict[str, float]
    parent_trial_id: str | None
    depth: int
    branch_id: str | None
    tags: dict[str, Any]


class ReadableState(Protocol):
    """Protocol for read-only access to system state.

    Used by policies to query current state without direct store access.
    """

    def capacity_free(self) -> int:
        """Get number of free capacity slots."""
        ...

    def trial(self, trial_id: str) -> TrialView | None:
        """Get a specific trial by ID."""
        ...

    def running_trials(self, experiment_id: str | None = None) -> list[TrialView]:
        """Get all running trials, optionally filtered by experiment."""
        ...

    def best_trials(
        self,
        experiment_id: str,
        top_n: int,
        key: str = "score",
        mode: Literal["min", "max"] = "max",
    ) -> list[TrialView]:
        """Get top N trials by a metric."""
        ...

    def trials_by_depth(self, experiment_id: str, depth: int) -> list[TrialView]:
        """Get all trials at a specific depth in the lineage tree."""
        ...

    def completed_trials(self, experiment_id: str) -> list[TrialView]:
        """Get all completed trials for an experiment."""
        ...

    def all_trials(self, experiment_id: str) -> list[TrialView]:
        """Get all trials regardless of status."""
        ...


class EventLog(Protocol):
    """Protocol for event log storage."""

    async def append(self, evt: Event) -> None:
        """Append an event to the log."""
        ...

    async def tail(self, n: int, *, aggregate_id: str | None = None) -> list[Event]:
        """Get last N events, optionally filtered by aggregate ID."""
        ...


class TrialStore(Protocol):
    """Protocol for trial storage."""

    def create(
        self, experiment_id: str, params: dict[str, Any], lineage: dict[str, Any]
    ) -> Trial:
        """Create a new trial."""
        ...

    def get(self, trial_id: str) -> Trial | None:
        """Get trial by ID."""
        ...

    def update(self, trial_id: str, **fields: Any) -> None:
        """Update trial fields."""
        ...

    def running(self, experiment_id: str | None = None) -> list[Trial]:
        """Get running trials."""
        ...

    def list_by_experiment(self, experiment_id: str) -> list[Trial]:
        """List all trials for a given experiment."""
        ...

    def best_of(
        self, experiment_id: str, metric: str, mode: Literal["min", "max"]
    ) -> dict[str, Any]:
        """Get best trial info by metric."""
        ...


class ExperimentStore(Protocol):
    """Protocol for experiment storage."""

    def create(self, spec: dict[str, Any]) -> Experiment:
        """Create a new experiment."""
        ...

    def get(self, experiment_id: str) -> Experiment | None:
        """Get experiment by ID."""
        ...

    def update(self, experiment_id: str, **fields: Any) -> None:
        """Update experiment fields."""
        ...


class DecisionStore(Protocol):
    """Protocol for decision audit storage."""

    def create(self, decision: Decision) -> None:
        """Store a new decision record."""
        ...

    def get(self, decision_id: str) -> Decision | None:
        """Get decision by ID."""
        ...

    def list_by_experiment(self, experiment_id: str) -> list[Decision]:
        """List all decisions for a given experiment."""
        ...


class Stores(Protocol):
    """Combined protocol for all stores."""

    events: EventLog
    trials: TrialStore
    experiments: ExperimentStore
    decisions: DecisionStore | None
