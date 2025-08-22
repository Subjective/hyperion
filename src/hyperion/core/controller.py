"""Central controller for command enforcement and trial scheduling."""

import logging
from collections import deque
from collections.abc import Callable
from typing import Any, Literal

from hyperion.core.bus import EventBus
from hyperion.core.capacity import CapacityManager
from hyperion.core.events import Command, CommandType, Event, EventType
from hyperion.core.executor import TrialExecutor
from hyperion.core.models import ExperimentStatus, TrialStatus
from hyperion.core.state import ReadableState, Stores, TrialView

logger = logging.getLogger(__name__)


class Controller:
    """Central controller that enforces commands and manages trial lifecycle.

    Subscribes to commands and lifecycle events, updating stores and
    coordinating with the executor and capacity manager.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        store: Stores,
        executor: TrialExecutor,
        capacity: CapacityManager,
        objective: Callable[..., Any] | None = None,
    ):
        """Initialize controller.

        Args:
            bus: Event bus for command/event communication
            store: Storage backends for experiments, trials, events
            executor: Trial executor implementation
            capacity: Capacity manager for admission control
            objective: Objective function to run
        """
        self.bus = bus
        self.store = store
        self.executor = executor
        self.capacity = capacity
        self.objective = objective

        # Queue for trials waiting for capacity
        self._queued_trials: deque[dict[str, Any]] = deque()

        # Subscribe to commands
        bus.subscribe(CommandType.START_EXPERIMENT, self._on_start_experiment)
        bus.subscribe(CommandType.START_TRIAL, self._on_start_trial)
        bus.subscribe(CommandType.KILL_TRIAL, self._on_kill_trial)
        bus.subscribe(CommandType.PATCH_TRIAL, self._on_patch_trial)

        # Subscribe to lifecycle events
        bus.subscribe(EventType.TRIAL_COMPLETED, self._on_trial_done)
        bus.subscribe(EventType.TRIAL_FAILED, self._on_trial_done)
        bus.subscribe(EventType.TRIAL_KILLED, self._on_trial_done)
        bus.subscribe(EventType.TRIAL_PROGRESS, self._on_trial_progress)

    # --- Command handlers ---

    async def _on_start_experiment(self, cmd: Command) -> None:
        """Handle START_EXPERIMENT command."""
        spec = cmd.data

        # Create experiment in store
        exp = self.store.experiments.create(spec)

        # Update status to running
        self.store.experiments.update(exp.id, status=ExperimentStatus.RUNNING)

        # Emit EXPERIMENT_STARTED event
        await self.bus.publish(
            Event(
                type=EventType.EXPERIMENT_STARTED,
                data={"experiment_id": exp.id, "name": exp.name},
                aggregate_id=exp.id,
                correlation_id=cmd.correlation_id,
                causation_id=cmd.id,
            )
        )

        logger.debug(f"Started experiment {exp.id}")

    async def _on_start_trial(self, cmd: Command) -> None:
        """Handle START_TRIAL command."""
        experiment_id = cmd.data["experiment_id"]
        params = cmd.data["params"]
        parent_trial_id = cmd.data.get("parent_trial_id")
        tags = cmd.data.get("tags", {})

        # Calculate lineage
        depth = 0
        if parent_trial_id:
            parent = self.store.trials.get(parent_trial_id)
            if parent:
                depth = parent.depth + 1

        lineage = {
            "parent_trial_id": parent_trial_id,
            "depth": depth,
            "branch_id": cmd.data.get("branch_id"),
            "mutation_op": cmd.data.get("mutation_op"),
        }

        # Create trial in store
        trial = self.store.trials.create(experiment_id, params, lineage)
        # Persist tags via store update for consistency across backends
        try:
            if tags:
                self.store.trials.update(trial.id, tags=tags)
        except Exception:
            logger.debug("Failed to persist trial tags via store.update; continuing")

        # Prepare correlation/causation metadata for downstream events
        meta: dict[str, Any] = {
            "correlation_id": cmd.correlation_id or cmd.id,
            "causation_id": cmd.id,
        }

        # Check capacity
        if self.capacity.can_admit(experiment_id):
            # Submit immediately
            await self._submit_trial(trial, meta)
        else:
            # Queue for later
            self._queued_trials.append(
                {
                    "trial": trial,
                    "meta": meta,
                }
            )
            logger.debug(f"Queued trial {trial.id} (at capacity)")

    async def _on_kill_trial(self, cmd: Command) -> None:
        """Handle KILL_TRIAL command."""
        trial_id = cmd.data["trial_id"]
        reason = cmd.data.get("reason", "")

        # Kill via executor with reason
        await self.executor.kill(trial_id, reason)

        logger.debug(f"Killed trial {trial_id}: {reason}")

    async def _on_patch_trial(self, cmd: Command) -> None:
        """Handle PATCH_TRIAL command."""
        trial_id = cmd.data["trial_id"]
        patch = cmd.data["patch"]

        # Apply patch via executor
        await self.executor.patch(trial_id, patch)

        logger.debug(f"Patched trial {trial_id} with {list(patch.keys())}")

    # --- Lifecycle handlers ---

    async def _on_trial_progress(self, evt: Event) -> None:
        """Handle trial progress events to update metrics."""
        trial_id = evt.data.get("trial_id")
        metrics = evt.data.get("metrics", {})

        if trial_id and metrics:
            # Update the trial's last reported metrics
            self.store.trials.update(trial_id, metrics_last=metrics)

    async def _on_trial_done(self, evt: Event) -> None:
        """Handle trial completion/failure/kill events."""
        trial_id = evt.data["trial_id"]
        trial = self.store.trials.get(trial_id)

        if not trial:
            logger.warning(f"Unknown trial {trial_id} in done event")
            return

        # Update trial status based on event type
        if evt.type == EventType.TRIAL_COMPLETED:
            self.store.trials.update(
                trial_id,
                status=TrialStatus.COMPLETED,
                score=evt.data.get("score"),
                metrics_last=evt.data.get("metrics", {}),
            )
        elif evt.type == EventType.TRIAL_FAILED:
            self.store.trials.update(trial_id, status=TrialStatus.FAILED)
        elif evt.type == EventType.TRIAL_KILLED:
            self.store.trials.update(trial_id, status=TrialStatus.KILLED)

        # Release capacity
        self.capacity.on_end(trial.experiment_id)

        # Emit CAPACITY_AVAILABLE (include aggregate and correlation for auditability)
        await self.bus.publish(
            Event(
                type=EventType.CAPACITY_AVAILABLE,
                data={"count": self.capacity.free_slots()},
                aggregate_id=trial.experiment_id,
                correlation_id=evt.correlation_id or evt.id,
                causation_id=evt.id,
            )
        )

        # Try to admit queued trials
        await self._process_queue()

    # --- Helper methods ---

    async def _submit_trial(
        self, trial: Any, meta: dict[str, Any] | None = None
    ) -> None:
        """Submit a trial to the executor."""
        # Update status to running
        # TODO: consider moving RUNNING transition to TRIAL_STARTED event handler
        # to perfectly align with lifecycle semantics.
        self.store.trials.update(trial.id, status=TrialStatus.RUNNING)

        # Record capacity usage
        self.capacity.on_start(trial.experiment_id)

        # Submit to executor
        if self.objective:
            await self.executor.submit(
                trial.id,
                trial.experiment_id,
                self.objective,
                trial.params,
                meta or {},
            )
        else:
            logger.error("No objective function provided to controller")

    async def _process_queue(self) -> None:
        """Process queued trials when capacity becomes available."""
        while self._queued_trials and self.capacity.free_slots() > 0:
            queued = self._queued_trials.popleft()
            trial = queued["trial"]
            meta = queued.get("meta")

            if self.capacity.can_admit(trial.experiment_id):
                await self._submit_trial(trial, meta)
            else:
                # Put back in queue if can't admit (shouldn't happen with free slots)
                self._queued_trials.appendleft(queued)
                break

    # --- State view ---

    def readable_state(self) -> ReadableState:
        """Get read-only view of current state."""
        return _ReadableStateAdapter(self.store, self.capacity)


class _ReadableStateAdapter(ReadableState):
    """Adapter that provides ReadableState interface over stores and capacity."""

    def __init__(self, store: Stores, capacity: CapacityManager):
        self.store = store
        self.capacity = capacity

    def capacity_free(self) -> int:
        """Get number of free capacity slots."""
        return self.capacity.free_slots()

    def trial(self, trial_id: str) -> TrialView | None:
        """Get a specific trial by ID."""
        t = self.store.trials.get(trial_id)
        if not t:
            return None

        return TrialView(
            trial_id=t.id,
            experiment_id=t.experiment_id,
            params=t.params,
            status=t.status.value,
            score=t.score,
            metrics_last=t.metrics_last,
            parent_trial_id=t.parent_trial_id,
            depth=t.depth,
            branch_id=t.branch_id,
            tags=t.tags,
        )

    def running_trials(self, experiment_id: str | None = None) -> list[TrialView]:
        """Get all running trials, optionally filtered by experiment."""
        trials = self.store.trials.running(experiment_id)
        return [
            TrialView(
                trial_id=t.id,
                experiment_id=t.experiment_id,
                params=t.params,
                status=t.status.value,
                score=t.score,
                metrics_last=t.metrics_last,
                parent_trial_id=t.parent_trial_id,
                depth=t.depth,
                branch_id=t.branch_id,
                tags=t.tags,
            )
            for t in trials
        ]

    def best_trials(
        self,
        experiment_id: str,
        top_n: int,
        key: str = "score",
        mode: Literal["min", "max"] = "max",
    ) -> list[TrialView]:
        """Get top N trials by a metric."""
        # Get all trials for the experiment
        all_trials = self.store.trials.list_by_experiment(experiment_id)

        # Filter to completed trials with scores
        completed_trials = [
            t
            for t in all_trials
            if t.status == TrialStatus.COMPLETED and t.score is not None
        ]

        # Sort by the specified metric
        if key == "score":
            sorted_trials = sorted(
                completed_trials,
                key=lambda t: t.score if t.score is not None else float("-inf"),
                reverse=(mode == "max"),
            )
        else:
            # Sort by a metric in metrics_last
            sorted_trials = sorted(
                completed_trials,
                key=lambda t: t.metrics_last.get(key, float("-inf")),
                reverse=(mode == "max"),
            )

        # Take top N and convert to TrialView
        return [
            TrialView(
                trial_id=t.id,
                experiment_id=t.experiment_id,
                params=t.params,
                status=t.status.value,
                score=t.score,
                metrics_last=t.metrics_last,
                parent_trial_id=t.parent_trial_id,
                depth=t.depth,
                branch_id=t.branch_id,
                tags=t.tags,
            )
            for t in sorted_trials[:top_n]
        ]

    def trials_by_depth(self, experiment_id: str, depth: int) -> list[TrialView]:
        """Get all trials at a specific depth in the lineage tree."""
        trials = self.store.trials.list_by_experiment(experiment_id)
        out: list[TrialView] = []
        for t in trials:
            if t.depth == depth:
                out.append(
                    TrialView(
                        trial_id=t.id,
                        experiment_id=t.experiment_id,
                        params=t.params,
                        status=t.status.value,
                        score=t.score,
                        metrics_last=t.metrics_last,
                        parent_trial_id=t.parent_trial_id,
                        depth=t.depth,
                        branch_id=t.branch_id,
                        tags=t.tags,
                    )
                )
        return out

    def completed_trials(self, experiment_id: str) -> list[TrialView]:
        """Get all completed trials for an experiment."""
        all_trials = self.store.trials.list_by_experiment(experiment_id)
        completed = [
            TrialView(
                trial_id=t.id,
                experiment_id=t.experiment_id,
                params=t.params,
                status=t.status.value,
                score=t.score,
                metrics_last=t.metrics_last,
                parent_trial_id=t.parent_trial_id,
                depth=t.depth,
                branch_id=t.branch_id,
                tags=t.tags,
            )
            for t in all_trials
            if t.status == TrialStatus.COMPLETED
        ]
        return completed

    def all_trials(self, experiment_id: str) -> list[TrialView]:
        """Get all trials regardless of status."""
        all_trials = self.store.trials.list_by_experiment(experiment_id)
        return [
            TrialView(
                trial_id=t.id,
                experiment_id=t.experiment_id,
                params=t.params,
                status=t.status.value,
                score=t.score,
                metrics_last=t.metrics_last,
                parent_trial_id=t.parent_trial_id,
                depth=t.depth,
                branch_id=t.branch_id,
                tags=t.tags,
            )
            for t in all_trials
        ]
