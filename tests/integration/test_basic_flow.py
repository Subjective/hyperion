"""Basic integration test to validate core architecture."""

import pytest

from hyperion.core.bus import InMemoryBus
from hyperion.core.capacity import CapacityManager
from hyperion.core.controller import Controller
from hyperion.core.effector import Effector
from hyperion.core.events import EventType
from hyperion.core.executor import LocalAsyncExecutor
from hyperion.core.models import ObjectiveResult, TrialStatus
from hyperion.core.state import Stores
from hyperion.storage.memory import (
    InMemoryEventLog,
    InMemoryExperimentStore,
    InMemoryTrialStore,
)


@pytest.mark.asyncio
async def test_end_to_end_with_controller_and_executor():
    """Wire InMemoryBus, stores, CapacityManager, LocalAsyncExecutor, and Controller."""
    # Setup services
    bus = InMemoryBus()
    event_log = InMemoryEventLog()
    trial_store = InMemoryTrialStore()
    experiment_store = InMemoryExperimentStore()

    class SimpleStores(Stores):
        def __init__(self):
            self.events = event_log
            self.trials = trial_store
            self.experiments = experiment_store
            self.decisions = None

    stores = SimpleStores()
    executor = LocalAsyncExecutor(bus)
    capacity = CapacityManager(max_concurrent=2)

    # Simple objective
    def objective(ctx, lr=0.01, batch_size=32):
        return ObjectiveResult(score=lr * batch_size)

    # Create controller
    Controller(
        bus=bus,
        store=stores,
        executor=executor,
        capacity=capacity,
        objective=objective,
    )

    # Track events
    received_events = []
    bus.subscribe("*", lambda e: received_events.append(e))

    # Create an experiment
    exp = experiment_store.create({"name": "test-experiment"})

    # Use effector to start a trial
    effector = Effector(bus)
    await effector.start_trial(
        experiment_id=exp.id, params={"lr": 0.01, "batch_size": 32}
    )

    # Wait for handlers to complete
    await bus.drain()

    # Verify trial was created
    trials = trial_store.list_by_experiment(exp.id)
    assert len(trials) == 1
    trial = trials[0]
    assert trial.experiment_id == exp.id
    assert trial.params["lr"] == 0.01

    # Trial should be either running or already completed (fast execution)
    assert trial.status in (TrialStatus.RUNNING, TrialStatus.COMPLETED)

    # Wait a bit for completion if still running
    if trial.status == TrialStatus.RUNNING:
        import asyncio

        await asyncio.sleep(0.1)
        await bus.drain()

    # Verify trial completed
    trial = trial_store.get(trial.id)
    assert trial is not None
    assert trial.status == TrialStatus.COMPLETED
    assert trial.score == 0.32  # 0.01 * 32

    # Verify lifecycle events were emitted
    event_types = [e.type for e in received_events]
    assert EventType.TRIAL_STARTED in event_types
    assert EventType.TRIAL_COMPLETED in event_types


@pytest.mark.asyncio
async def test_lineage_tracking_with_controller():
    """Parent then child trial (via parent_trial_id); assert lineage fields in store."""
    # Setup services
    bus = InMemoryBus()
    event_log = InMemoryEventLog()
    trial_store = InMemoryTrialStore()
    experiment_store = InMemoryExperimentStore()

    class SimpleStores(Stores):
        def __init__(self):
            self.events = event_log
            self.trials = trial_store
            self.experiments = experiment_store
            self.decisions = None

    stores = SimpleStores()
    executor = LocalAsyncExecutor(bus)
    capacity = CapacityManager(max_concurrent=2)

    # Simple objective
    def objective(ctx, **params):
        return ObjectiveResult(score=0.5)

    # Create controller
    Controller(
        bus=bus,
        store=stores,
        executor=executor,
        capacity=capacity,
        objective=objective,
    )

    # Create experiment
    exp = experiment_store.create({"name": "test-lineage"})

    # Create parent trial
    effector = Effector(bus)
    await effector.start_trial(exp.id, {"lr": 0.01})
    await bus.drain()

    parent = trial_store.list_by_experiment(exp.id)[0]
    assert parent.parent_trial_id is None
    assert parent.depth == 0

    # Create child trial with parent reference
    await effector.start_trial(exp.id, {"lr": 0.001}, parent_trial_id=parent.id)
    await bus.drain()

    trials = trial_store.list_by_experiment(exp.id)
    assert len(trials) == 2

    child = trials[1]
    assert child.parent_trial_id == parent.id
    assert child.depth == 1  # Parent depth + 1
    assert parent.depth == 0
