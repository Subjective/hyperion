"""Framework layer for experiment orchestration."""

import asyncio
import builtins
import contextlib
import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from hyperion.core.controller import Controller
from hyperion.core.effector import Effector
from hyperion.core.events import Event, EventType
from hyperion.core.models import (
    Decision,
    ExperimentStatus,
    ObjectiveResult,
)
from hyperion.core.state import Stores
from hyperion.framework.policy import KillTrial, PatchTrial, Policy, StartTrial

logger = logging.getLogger(__name__)


@dataclass
class Resources:
    """Runtime constraints for scheduling and capacity."""

    max_concurrent: int = 4
    quotas: dict[str, int] = field(default_factory=lambda: dict[str, int]())
    weights: dict[str, float] = field(default_factory=lambda: dict[str, float]())
    timeout_s: int | None = None


@dataclass
class Budget:
    """Search limits and stopping conditions."""

    max_trials: int
    max_time_s: int | None = None
    metric: str = "score"
    mode: Literal["min", "max"] = "max"
    target: float | None = None


@dataclass
class Monitoring:
    """Toggle integrations and attach user callbacks."""

    callbacks: list[Any] = field(default_factory=lambda: list[Any]())


@dataclass
class Pipeline:
    """A list of orchestration components; typically Policies."""

    steps: Sequence[Policy]


@dataclass
class ExperimentSpec:
    """Specification for a hyperparameter optimization experiment."""

    name: str
    objective: Callable[..., ObjectiveResult]
    search_space: dict[str, Any]
    pipeline: Pipeline
    resources: Resources = field(default_factory=Resources)
    monitoring: Monitoring | dict[str, Any] = field(default_factory=Monitoring)
    budget: Budget = field(default_factory=lambda: Budget(max_trials=50))
    tags: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())


class ExperimentRunner:
    """Wires the spec to core services and executes the run.

    Creates controller, hosts policies via bus subscriptions,
    and drives policy decisions on relevant triggers.
    """

    def __init__(self, spec: ExperimentSpec, *, services: dict[str, Any]):
        """Initialize runner.

        Args:
            spec: Experiment specification
            services: Dict with keys: {"bus", "stores", "executor", "capacity"}
        """
        self.spec = spec
        self.services = services
        self.experiment_id: str | None = None
        self._stop_requested = False
        self._policy_tasks: list[asyncio.Task[None]] = []
        # Metadata tracking for correlation/causation
        self._start_cmd_id: str | None = None
        self._last_terminal_event_id: str | None = None
        # Track trials started for authoritative budget enforcement
        self._trials_started_count = 0

    async def run(self) -> dict[str, Any]:
        """Run the experiment to completion.

        Returns:
            Dict with experiment name and best trial info
        """
        bus = self.services["bus"]
        stores = self.services["stores"]
        executor = self.services["executor"]
        capacity = self.services["capacity"]

        # Configure capacity from Resources
        capacity.max_concurrent = self.spec.resources.max_concurrent

        # Subscribe to all events and persist them to EventLog
        async def persist_event(evt: Event) -> None:
            """Store event in the EventLog."""
            await stores.events.append(evt)

        bus.subscribe("*", persist_event)

        # Create controller with objective
        controller = Controller(
            bus=bus,
            store=stores,
            executor=executor,
            capacity=capacity,
            objective=self.spec.objective,
        )

        # Subscribe callbacks
        monitoring = self.spec.monitoring
        callbacks = (
            monitoring.get("callbacks", [])
            if isinstance(monitoring, dict)
            else monitoring.callbacks
        )

        for cb in callbacks:
            bus.subscribe("*", cb.on_event)

        # Prepare to capture experiment_id and completion
        self._exp_started = asyncio.Event()
        self._exp_completed = asyncio.Event()

        def _capture_exp_started(evt: Event):
            if evt.type == EventType.EXPERIMENT_STARTED and not self.experiment_id:
                with contextlib.suppress(Exception):
                    self.experiment_id = evt.data["experiment_id"]
                    # Record the causation id (id of START_EXPERIMENT command)
                    self._start_cmd_id = evt.causation_id
                    self._exp_started.set()

        def _capture_exp_completed(evt: Event):
            if evt.type == EventType.EXPERIMENT_COMPLETED:
                self._exp_completed.set()

        bus.subscribe(EventType.EXPERIMENT_STARTED, _capture_exp_started)
        bus.subscribe(EventType.EXPERIMENT_COMPLETED, _capture_exp_completed)

        # Host policies (they will receive EXPERIMENT_STARTED via forward_events)
        for step in self.spec.pipeline.steps:
            await self._host_policy(step, controller)

        # Start the experiment
        eff = Effector(bus)
        await eff.start_experiment(
            {
                "name": self.spec.name,
                "config": {
                    "search_space": self.spec.search_space,
                    "budget": {
                        "max_trials": self.spec.budget.max_trials,
                        "max_time_s": self.spec.budget.max_time_s,
                        "metric": self.spec.budget.metric,
                        "mode": self.spec.budget.mode,
                        "target": self.spec.budget.target,
                    },
                    "resources": {
                        "max_concurrent": self.spec.resources.max_concurrent,
                        "quotas": self.spec.resources.quotas,
                        "weights": self.spec.resources.weights,
                        "timeout_s": self.spec.resources.timeout_s,
                    },
                },
                "tags": self.spec.tags,
            }
        )

        # Await EXPERIMENT_STARTED and set experiment_id
        try:
            await asyncio.wait_for(self._exp_started.wait(), timeout=5.0)
        except TimeoutError:
            logger.warning(
                "Timed out waiting for EXPERIMENT_STARTED; "
                "some policies may not receive experiment_id"
            )

        # Update policies with experiment ID
        for step in self.spec.pipeline.steps:
            step.experiment_id = self.experiment_id

        # Track last terminal trial event id for causation
        def _track_terminal(evt: Event):
            if evt.type in (
                EventType.TRIAL_COMPLETED,
                EventType.TRIAL_FAILED,
                EventType.TRIAL_KILLED,
            ):
                self._last_terminal_event_id = evt.id

        bus.subscribe("*", _track_terminal)

        # Wait for completion or timeout
        try:
            await self._wait_for_completion(stores)
        except TimeoutError:
            logger.warning(f"Experiment {self.spec.name} timed out")

        # Cancel policy tasks
        for task in self._policy_tasks:
            task.cancel()

        # Ensure all event handlers have drained (important for tests)
        with contextlib.suppress(Exception):
            await self.services["bus"].drain()

        # Return best result
        if self.experiment_id:
            best = stores.trials.best_of(
                self.experiment_id,
                metric=self.spec.budget.metric,
                mode=self.spec.budget.mode,
            )
        else:
            best = {}

        return {
            "experiment": self.spec.name,
            "best": best,
        }

    async def _host_policy(self, policy: Policy, controller: Controller) -> None:
        """Host a policy by wiring it to the bus and driving decisions.

        Args:
            policy: Policy instance to host
            controller: Controller instance for readable state
        """
        bus = self.services["bus"]
        eff = Effector(bus)

        # Forward all events to the policy
        if hasattr(policy, "on_events"):

            async def forward_events(evt: Event):
                await policy.on_events([evt])

            bus.subscribe("*", forward_events)

        # Drive decisions on triggers
        triggers = (
            EventType.TRIAL_COMPLETED,
            EventType.CAPACITY_AVAILABLE,
            EventType.EXPERIMENT_STARTED,
            EventType.TRIAL_PROGRESS,
        )

        async def driver():
            """Drive policy decisions on trigger events."""
            event_queue: asyncio.Queue[Event] = asyncio.Queue()

            # Subscribe to trigger events
            def queue_event(evt: Event):
                if evt.type in triggers:
                    with contextlib.suppress(builtins.BaseException):
                        event_queue.put_nowait(evt)

            bus.subscribe("*", queue_event)

            # Wait for triggers and drive decisions
            while not self._stop_requested:
                try:
                    # Wait for a trigger event
                    await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    # TODO: consider a sentinel/stop event or an EventBus-based
                    # awaitable to avoid timeouts for loop exit conditions.

                    # Get current state and ask for decisions
                    # Gate decisions until experiment_id is available
                    if not self.experiment_id:
                        continue
                    state = controller.readable_state()
                    actions = await policy.decide(state)

                    # Execute actions and capture command_ids to link decisions → trials
                    recorded: list[tuple[Any, str | None]] = []
                    for action in actions:
                        if isinstance(action, StartTrial):
                            # Enforce budget at framework layer with authoritative counter
                            if (
                                self._trials_started_count
                                >= self.spec.budget.max_trials
                            ):
                                logger.debug(
                                    f"Blocking StartTrial - budget exhausted "
                                    f"({self._trials_started_count}/{self.spec.budget.max_trials} trials started)"
                                )
                                continue

                            cmd_id = await eff.start_trial(
                                action.experiment_id,
                                action.params,
                                action.parent_trial_id,
                                action.tags,
                            )
                            self._trials_started_count += 1
                            recorded.append((action, cmd_id))
                        elif isinstance(action, KillTrial):
                            cmd_id = await eff.kill_trial(
                                action.trial_id, action.reason
                            )
                            recorded.append((action, cmd_id))
                        else:  # PatchTrial
                            cmd_id = await eff.patch_trial(
                                action.trial_id, action.patch
                            )
                            recorded.append((action, cmd_id))
                        # TODO: insert a simple arbiter hook here if multiple
                        # policies are hosted concurrently (priority/quotas).

                    # Record decision if actions were taken
                    if recorded:
                        await self._record_decision(policy, recorded)

                except TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in policy driver: {e}")

        # Start driver task
        task = asyncio.create_task(driver())
        self._policy_tasks.append(task)

    async def _wait_for_completion(self, stores: Stores) -> None:
        """Wait for experiment to complete based on budget.

        Args:
            stores: Storage backends
        """
        max_wait = self.spec.budget.max_time_s  # None means no time cap
        check_interval = 0.1
        elapsed = 0.0

        while True:
            await asyncio.sleep(check_interval)
            if max_wait is not None:
                elapsed += check_interval
                if elapsed >= max_wait:
                    break

            if not self.experiment_id:
                continue

            # Event-based completion
            if self._exp_completed.is_set():
                break

            # Check if we've hit max_trials
            exp_trials = stores.trials.list_by_experiment(self.experiment_id)

            if len(exp_trials) >= self.spec.budget.max_trials:
                # Check if all are completed
                from hyperion.core.models import TrialStatus

                completed_statuses = {
                    TrialStatus.COMPLETED,
                    TrialStatus.FAILED,
                    TrialStatus.KILLED,
                }

                if all(t.status in completed_statuses for t in exp_trials):
                    logger.debug(f"All {len(exp_trials)} trials completed")
                    break

            # Check for target reached
            if self.spec.budget.target is not None:
                best = stores.trials.best_of(
                    self.experiment_id,
                    metric=self.spec.budget.metric,
                    mode=self.spec.budget.mode,
                )
                if best:
                    if self.spec.budget.metric == "score":
                        value = best.get("score")
                    else:
                        value = best.get("metrics", {}).get(self.spec.budget.metric)
                    if (
                        self.spec.budget.mode == "max"
                        and value is not None
                        and value >= self.spec.budget.target
                    ):
                        logger.debug(
                            f"Target reached: {value} >= {self.spec.budget.target}"
                        )
                        break
                    elif (
                        self.spec.budget.mode == "min"
                        and value is not None
                        and value <= self.spec.budget.target
                    ):
                        logger.debug(
                            f"Target reached: {value} <= {self.spec.budget.target}"
                        )
                        break

        # Signal stop to policy drivers
        self._stop_requested = True

        # Emit EXPERIMENT_COMPLETED and update experiment status in store
        try:
            if self.experiment_id:
                # Update experiment status
                with contextlib.suppress(Exception):
                    stores.experiments.update(
                        self.experiment_id, status=ExperimentStatus.COMPLETED
                    )

                # Publish completion event
                await self.services["bus"].publish(
                    Event(
                        type=EventType.EXPERIMENT_COMPLETED,
                        data={"experiment_id": self.experiment_id},
                        aggregate_id=self.experiment_id,
                        correlation_id=getattr(self, "_start_cmd_id", None),
                        causation_id=getattr(self, "_last_terminal_event_id", None),
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to emit experiment completion: {e}")

    async def _record_decision(
        self, policy: Policy, actions_with_ids: list[tuple[Any, str | None]]
    ) -> None:
        """Record a decision made by a policy.

        Args:
            policy: The policy that made the decision
            actions_with_ids: List of (action, command_id) pairs
        """
        if not self.experiment_id:
            return

        # Get rationale from the policy
        try:
            rationale = await policy.rationale()
        except Exception as e:
            logger.warning(
                f"Failed to get rationale from {policy.__class__.__name__}: {e}"
            )
            rationale = None

        # Serialize actions for storage
        serialized_actions: list[dict[str, Any]] = []
        for action, cmd_id in actions_with_ids:
            if isinstance(action, StartTrial):
                serialized_actions.append(
                    {
                        "type": "StartTrial",
                        "experiment_id": action.experiment_id,
                        "params": action.params,
                        "parent_trial_id": action.parent_trial_id,
                        "tags": action.tags,
                        "command_id": cmd_id,
                    }
                )
            elif isinstance(action, KillTrial):
                serialized_actions.append(
                    {
                        "type": "KillTrial",
                        "trial_id": action.trial_id,
                        "reason": action.reason,
                        "command_id": cmd_id,
                    }
                )
            elif isinstance(action, PatchTrial):
                serialized_actions.append(
                    {
                        "type": "PatchTrial",
                        "trial_id": action.trial_id,
                        "patch": action.patch,
                        "command_id": cmd_id,
                    }
                )

        # Create decision record
        decision = Decision(
            id=f"decision-{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(UTC),
            experiment_id=self.experiment_id,
            actor_type="strategy" if "Policy" in policy.__class__.__name__ else "agent",
            actor_id=policy.__class__.__name__,
            actions=serialized_actions,
            rationale=rationale,
        )

        # Store decision if DecisionStore is available
        stores = self.services.get("stores")
        if stores and hasattr(stores, "decisions") and stores.decisions:
            stores.decisions.create(decision)

        # Emit DECISION_RECORDED event
        bus = self.services["bus"]
        await bus.publish(
            Event(
                type=EventType.DECISION_RECORDED,
                data={
                    "decision_id": decision.id,
                    "experiment_id": decision.experiment_id,
                    "actor_type": decision.actor_type,
                    "actor_id": decision.actor_id,
                    "actions": decision.actions,
                    "rationale": decision.rationale,
                },
                aggregate_id=self.experiment_id,
            )
        )
