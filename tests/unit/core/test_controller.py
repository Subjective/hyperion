"""Tests for the controller component."""

import pytest

from hyperion.core.bus import InMemoryBus
from hyperion.core.capacity import CapacityManager
from hyperion.core.controller import Controller
from hyperion.core.events import Command, CommandType, Event, EventType
from hyperion.core.models import ObjectiveResult, TrialStatus
from hyperion.core.state import Stores
from hyperion.storage.memory import (
    InMemoryEventLog,
    InMemoryExperimentStore,
    InMemoryTrialStore,
)


class SimpleStores(Stores):
    """Simple stores wrapper for testing that implements Stores protocol."""

    def __init__(self):
        self.events = InMemoryEventLog()
        self.trials = InMemoryTrialStore()
        self.experiments = InMemoryExperimentStore()
        self.decisions = None


@pytest.mark.asyncio
async def test_start_trial_updates_store_and_emits_started():
    """On START_TRIAL, creates a Trial, emits TRIAL_STARTED, and calls executor submit."""
    bus = InMemoryBus()
    store = SimpleStores()

    # Create a mock executor that tracks submissions
    submitted_trials = []

    class MockExecutor:
        async def submit(self, trial_id, experiment_id, objective, params, meta=None):
            submitted_trials.append(trial_id)
            # Emit TRIAL_STARTED like real executor would
            await bus.publish(
                Event(
                    type=EventType.TRIAL_STARTED,
                    data={"trial_id": trial_id, "params": params},
                    aggregate_id=experiment_id,
                )
            )

        async def kill(self, trial_id, reason=""):
            pass

        async def patch(self, trial_id, patch):
            pass

    executor = MockExecutor()
    capacity = CapacityManager(max_concurrent=2)

    # Track events
    events = []
    bus.subscribe("*", lambda e: events.append(e))

    # Create experiment first
    exp = store.experiments.create({"name": "test-exp"})

    # Create controller with objective
    def dummy_objective(ctx, **params):
        return ObjectiveResult(score=0.5)

    Controller(
        bus=bus,
        store=store,
        executor=executor,
        capacity=capacity,
        objective=dummy_objective,
    )

    # Send START_TRIAL command
    await bus.publish(
        Command(
            type=CommandType.START_TRIAL,
            data={
                "experiment_id": exp.id,
                "params": {"lr": 0.01},
                "parent_trial_id": None,
                "tags": {"strategy": "random", "note": "test"},
            },
        )
    )
    await bus.drain()

    # Check trial was created in store
    trials = store.trials.list_by_experiment(exp.id)
    assert len(trials) == 1
    trial = trials[0]
    assert trial.experiment_id == exp.id
    assert trial.params["lr"] == 0.01
    assert trial.status == TrialStatus.RUNNING
    # Tags should be persisted via the store
    assert trial.tags.get("strategy") == "random"
    assert trial.tags.get("note") == "test"

    # Check executor was called
    assert len(submitted_trials) == 1
    assert submitted_trials[0] == trial.id

    # Check TRIAL_STARTED event was emitted
    started_events = [e for e in events if e.type == EventType.TRIAL_STARTED]
    assert len(started_events) == 1
    assert started_events[0].data["trial_id"] == trial.id


@pytest.mark.asyncio
async def test_capacity_limits_and_queue_release():
    """With max_concurrent=1, second START_TRIAL is deferred and admitted after first completes."""
    bus = InMemoryBus()
    store = SimpleStores()

    # Track which trials get submitted
    submitted_trials = []

    class MockExecutor:
        async def submit(self, trial_id, experiment_id, objective, params, meta=None):
            submitted_trials.append(trial_id)
            await bus.publish(
                Event(
                    type=EventType.TRIAL_STARTED,
                    data={"trial_id": trial_id},
                    aggregate_id=experiment_id,
                )
            )

        async def kill(self, trial_id, reason=""):
            pass

        async def patch(self, trial_id, patch):
            pass

    executor = MockExecutor()
    capacity = CapacityManager(max_concurrent=1)  # Only 1 at a time

    # Track events
    events = []
    bus.subscribe("*", lambda e: events.append(e))

    # Create experiment
    exp = store.experiments.create({"name": "test-exp"})

    # Create controller
    def dummy_objective(ctx, **params):
        return ObjectiveResult(score=0.5)

    Controller(
        bus=bus,
        store=store,
        executor=executor,
        capacity=capacity,
        objective=dummy_objective,
    )

    # Send two START_TRIAL commands
    await bus.publish(
        Command(
            type=CommandType.START_TRIAL,
            data={"experiment_id": exp.id, "params": {"id": 1}},
        )
    )
    await bus.publish(
        Command(
            type=CommandType.START_TRIAL,
            data={"experiment_id": exp.id, "params": {"id": 2}},
        )
    )
    await bus.drain()

    # Only first should be submitted due to capacity
    assert len(submitted_trials) == 1
    first_trial_id = submitted_trials[0]

    # Check that second trial is queued (pending)
    trials = store.trials.list_by_experiment(exp.id)
    assert len(trials) == 2
    running = [t for t in trials if t.status == TrialStatus.RUNNING]
    pending = [t for t in trials if t.status == TrialStatus.PENDING]
    assert len(running) == 1
    assert len(pending) == 1

    # Complete the first trial
    await bus.publish(
        Event(
            type=EventType.TRIAL_COMPLETED,
            data={"trial_id": first_trial_id, "score": 0.9},
            aggregate_id=exp.id,
        )
    )
    await bus.drain()

    # Second trial should now be submitted
    assert len(submitted_trials) == 2

    # Check CAPACITY_AVAILABLE was emitted
    capacity_events = [e for e in events if e.type == EventType.CAPACITY_AVAILABLE]
    assert len(capacity_events) >= 1


@pytest.mark.asyncio
async def test_controller_handles_kill_trial_command():
    """Test that KILL_TRIAL command calls executor.kill()."""
    bus = InMemoryBus()
    store = SimpleStores()

    # Track killed trials
    killed_trials = []

    class MockExecutor:
        async def submit(self, trial_id, experiment_id, objective, params, meta=None):
            await bus.publish(
                Event(
                    type=EventType.TRIAL_STARTED,
                    data={"trial_id": trial_id},
                    aggregate_id=experiment_id,
                )
            )

        async def kill(self, trial_id, reason=""):
            killed_trials.append(trial_id)
            # Emit TRIAL_KILLED like real executor would
            await bus.publish(
                Event(
                    type=EventType.TRIAL_KILLED,
                    data={"trial_id": trial_id, "reason": reason or "User requested"},
                    aggregate_id=exp.id,
                )
            )

        async def patch(self, trial_id, patch):
            pass

    executor = MockExecutor()
    capacity = CapacityManager(max_concurrent=2)

    # Track events
    events = []
    bus.subscribe("*", lambda e: events.append(e))

    # Create experiment and trial
    exp = store.experiments.create({"name": "test-exp"})
    trial = store.trials.create(exp.id, {"lr": 0.01}, {})
    store.trials.update(trial.id, status=TrialStatus.RUNNING)

    # Create controller
    def dummy_objective(ctx, **params):
        return ObjectiveResult(score=0.5)

    Controller(
        bus=bus,
        store=store,
        executor=executor,
        capacity=capacity,
        objective=dummy_objective,
    )

    # Send KILL_TRIAL command
    await bus.publish(
        Command(
            type=CommandType.KILL_TRIAL,
            data={"trial_id": trial.id, "reason": "Early stopping"},
        )
    )
    await bus.drain()

    # Check executor.kill was called
    assert len(killed_trials) == 1
    assert killed_trials[0] == trial.id

    # Check TRIAL_KILLED event was emitted
    killed_events = [e for e in events if e.type == EventType.TRIAL_KILLED]
    assert len(killed_events) == 1
    assert killed_events[0].data["trial_id"] == trial.id

    # Check trial status was updated to KILLED
    updated_trial = store.trials.get(trial.id)
    assert updated_trial is not None
    assert updated_trial.status == TrialStatus.KILLED


@pytest.mark.asyncio
async def test_controller_handles_patch_trial_command():
    """Test that PATCH_TRIAL command calls executor.patch()."""
    bus = InMemoryBus()
    store = SimpleStores()

    # Track patched trials
    patched_trials = []

    class MockExecutor:
        async def submit(self, trial_id, experiment_id, objective, params, meta=None):
            pass

        async def kill(self, trial_id, reason=""):
            pass

        async def patch(self, trial_id, patch):
            patched_trials.append((trial_id, patch))

    executor = MockExecutor()
    capacity = CapacityManager(max_concurrent=2)

    # Create experiment and trial
    exp = store.experiments.create({"name": "test-exp"})
    trial = store.trials.create(exp.id, {"lr": 0.01}, {})
    store.trials.update(trial.id, status=TrialStatus.RUNNING)

    # Create controller
    def dummy_objective(ctx, **params):
        return ObjectiveResult(score=0.5)

    Controller(
        bus=bus,
        store=store,
        executor=executor,
        capacity=capacity,
        objective=dummy_objective,
    )

    # Send PATCH_TRIAL command
    patch_data = {"lr": 0.001, "batch_size": 64}
    await bus.publish(
        Command(
            type=CommandType.PATCH_TRIAL,
            data={"trial_id": trial.id, "patch": patch_data},
        )
    )
    await bus.drain()

    # Check executor.patch was called
    assert len(patched_trials) == 1
    assert patched_trials[0][0] == trial.id
    assert patched_trials[0][1] == patch_data


@pytest.mark.asyncio
async def test_controller_handles_unknown_trial_in_done_event():
    """Test that controller handles done event for unknown trial gracefully."""
    bus = InMemoryBus()
    store = SimpleStores()

    class MockExecutor:
        async def submit(self, trial_id, experiment_id, objective, params, meta=None):
            pass

        async def kill(self, trial_id, reason=""):
            pass

        async def patch(self, trial_id, patch):
            pass

    executor = MockExecutor()
    capacity = CapacityManager(max_concurrent=2)

    # Track events
    events = []
    bus.subscribe("*", lambda e: events.append(e))

    # Create controller
    def dummy_objective(ctx, **params):
        return ObjectiveResult(score=0.5)

    Controller(
        bus=bus,
        store=store,
        executor=executor,
        capacity=capacity,
        objective=dummy_objective,
    )

    # Send TRIAL_COMPLETED for non-existent trial
    await bus.publish(
        Event(
            type=EventType.TRIAL_COMPLETED,
            data={"trial_id": "unknown-trial-id", "score": 0.9},
            aggregate_id="unknown-trial-id",
        )
    )
    await bus.drain()

    # Should not crash, just log warning
    # No CAPACITY_AVAILABLE should be emitted for unknown trial
    capacity_events = [e for e in events if e.type == EventType.CAPACITY_AVAILABLE]
    assert len(capacity_events) == 0
