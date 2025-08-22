"""In-memory storage implementations for development and testing."""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from hyperion.core.events import Event
from hyperion.core.models import (
    Decision,
    Experiment,
    ExperimentStatus,
    Trial,
    TrialStatus,
)
from hyperion.core.state import DecisionStore, EventLog, ExperimentStore, TrialStore

logger = logging.getLogger(__name__)


class InMemoryEventLog(EventLog):
    """In-memory event log implementation."""

    def __init__(self):
        self._events: list[Event] = []

    async def append(self, evt: Event) -> None:
        """Append event to the log."""
        self._events.append(evt)
        logger.debug(
            f"Appended event {evt.type} to log, total events: {len(self._events)}"
        )

    async def tail(self, n: int, *, aggregate_id: str | None = None) -> list[Event]:
        """Get last N events, optionally filtered by aggregate ID."""
        events = self._events

        if aggregate_id is not None:
            events = [e for e in events if e.aggregate_id == aggregate_id]

        # Return last n events
        return events[-n:] if len(events) > n else events


class InMemoryTrialStore(TrialStore):
    """In-memory trial storage implementation."""

    def __init__(self):
        self._trials: dict[str, Trial] = {}

    def create(
        self, experiment_id: str, params: dict[str, Any], lineage: dict[str, Any]
    ) -> Trial:
        """Create a new trial."""
        trial_id = f"trial-{uuid.uuid4().hex[:8]}"

        trial = Trial(
            id=trial_id,
            experiment_id=experiment_id,
            params=params,
            parent_trial_id=lineage.get("parent_trial_id"),
            depth=lineage.get("depth", 0),
            branch_id=lineage.get("branch_id"),
            mutation_op=lineage.get("mutation_op"),
        )

        self._trials[trial_id] = trial
        logger.debug(f"Created trial {trial_id} for experiment {experiment_id}")
        return trial

    def get(self, trial_id: str) -> Trial | None:
        """Get trial by ID."""
        return self._trials.get(trial_id)

    def update(self, trial_id: str, **fields: Any) -> None:
        """Update trial fields."""
        trial = self._trials.get(trial_id)
        if trial:
            for key, value in fields.items():
                if hasattr(trial, key):
                    setattr(trial, key, value)
            logger.debug(f"Updated trial {trial_id} with {list(fields.keys())}")
        else:
            logger.warning(f"Attempted to update non-existent trial {trial_id}")

    def running(self, experiment_id: str | None = None) -> list[Trial]:
        """Get running trials, optionally filtered by experiment."""
        running = [t for t in self._trials.values() if t.status == TrialStatus.RUNNING]

        if experiment_id:
            running = [t for t in running if t.experiment_id == experiment_id]

        return running

    def list_by_experiment(self, experiment_id: str) -> list[Trial]:
        """List all trials for a given experiment."""
        return [t for t in self._trials.values() if t.experiment_id == experiment_id]

    def best_of(
        self, experiment_id: str, metric: str, mode: Literal["min", "max"]
    ) -> dict[str, Any]:
        """Find best trial by metric.

        Searches for the best trial by either the 'score' field or any metric
        in 'metrics_last'. Returns empty dict if no trials have the metric.
        """
        # Get completed trials for this experiment
        completed = [
            t
            for t in self._trials.values()
            if t.experiment_id == experiment_id and t.status == TrialStatus.COMPLETED
        ]

        if not completed:
            return {}

        # Helper to extract metric value uniformly
        def get_metric(trial: Trial) -> float | None:
            """Get metric value from either score field or metrics_last."""
            if metric == "score":
                return trial.score
            return trial.metrics_last.get(metric)

        # Filter to trials that have this metric
        trials_with_metric = [t for t in completed if get_metric(t) is not None]

        if not trials_with_metric:
            return {}

        # Find the best trial
        if mode == "max":
            best = max(trials_with_metric, key=lambda t: get_metric(t) or float("-inf"))
        else:  # min
            best = min(trials_with_metric, key=lambda t: get_metric(t) or float("inf"))

        # Return consistent structure with all trial information
        return {
            "trial_id": best.id,
            "params": best.params,
            "score": best.score,
            "metrics": best.metrics_last,
        }


class InMemoryExperimentStore(ExperimentStore):
    """In-memory experiment storage implementation."""

    def __init__(self):
        self._experiments: dict[str, Experiment] = {}

    def create(self, spec: dict[str, Any]) -> Experiment:
        """Create a new experiment."""
        exp_id = f"exp-{uuid.uuid4().hex[:8]}"

        exp = Experiment(
            id=exp_id,
            name=spec.get("name", "unnamed"),
            created_at=datetime.now(UTC),
            status=ExperimentStatus.PENDING,
            config=spec.get("config", {}),
            tags=spec.get("tags", {}),
        )

        self._experiments[exp_id] = exp
        logger.debug(f"Created experiment {exp_id} with name '{exp.name}'")
        return exp

    def get(self, experiment_id: str) -> Experiment | None:
        """Get experiment by ID."""
        return self._experiments.get(experiment_id)

    def update(self, experiment_id: str, **fields: Any) -> None:
        """Update experiment fields."""
        exp = self._experiments.get(experiment_id)
        if exp:
            for key, value in fields.items():
                if hasattr(exp, key):
                    setattr(exp, key, value)
            logger.debug(
                f"Updated experiment {experiment_id} with {list(fields.keys())}"
            )
        else:
            logger.warning(
                f"Attempted to update non-existent experiment {experiment_id}"
            )


class InMemoryDecisionStore(DecisionStore):
    """In-memory decision audit storage implementation."""

    def __init__(self):
        self._decisions: dict[str, Decision] = {}

    def create(self, decision: Decision) -> None:
        """Store a new decision record."""
        self._decisions[decision.id] = decision
        logger.debug(
            f"Stored decision {decision.id} by {decision.actor_id} "
            f"for experiment {decision.experiment_id}"
        )

    def get(self, decision_id: str) -> Decision | None:
        """Get decision by ID."""
        return self._decisions.get(decision_id)

    def list_by_experiment(self, experiment_id: str) -> list[Decision]:
        """List all decisions for a given experiment."""
        return [d for d in self._decisions.values() if d.experiment_id == experiment_id]


class InMemoryStores:
    """Combined in-memory storage implementation."""

    def __init__(self):
        """Initialize in-memory stores."""
        self.events = InMemoryEventLog()
        self.trials = InMemoryTrialStore()
        self.experiments = InMemoryExperimentStore()
        self.decisions = InMemoryDecisionStore()
