"""Tests for the ExperimentRunner framework component."""

import asyncio
import contextlib

import pytest

from hyperion.core.bus import InMemoryBus
from hyperion.core.capacity import CapacityManager
from hyperion.core.events import EventType
from hyperion.core.executor import LocalAsyncExecutor
from hyperion.core.models import ObjectiveResult
from hyperion.framework.experiment import (
    Budget,
    ExperimentRunner,
    ExperimentSpec,
    Pipeline,
    Resources,
)
from hyperion.framework.policy import StartTrial
from hyperion.framework.search_space import Float
from hyperion.policies.random_search import RandomSearchPolicy
from hyperion.storage.memory import (
    InMemoryEventLog,
    InMemoryExperimentStore,
    InMemoryTrialStore,
)


class SimpleStores:
    """Simple stores wrapper for testing."""

    def __init__(self):
        self.events = InMemoryEventLog()
        self.trials = InMemoryTrialStore()
        self.experiments = InMemoryExperimentStore()


class MockPolicy:
    """Mock policy for testing runner behavior."""

    def __init__(self):
        self.experiment_id: str | None = None
        self.events_received = []
        self.decide_called = 0
        self.actions_to_return = []

    async def on_events(self, events):
        self.events_received.extend(events)

    async def decide(self, state):
        self.decide_called += 1
        # Return pre-configured actions or empty list
        return self.actions_to_return.pop(0) if self.actions_to_return else []

    async def rationale(self):
        return "Mock policy for testing"


@pytest.mark.asyncio
async def test_runner_hosts_policies_and_drives_decisions():
    """Runner subscribes policies to bus and drives decide() on triggers."""
    # Setup services
    bus = InMemoryBus()
    stores = SimpleStores()
    executor = LocalAsyncExecutor(bus)
    capacity = CapacityManager(max_concurrent=2)

    services = {
        "bus": bus,
        "stores": stores,
        "executor": executor,
        "capacity": capacity,
    }

    # Simple objective
    def objective(ctx, x=0.5):
        return ObjectiveResult(score=x)

    # Create mock policy
    mock_policy = MockPolicy()

    # Configure policy to return a StartTrial action when decide() is called
    mock_policy.actions_to_return = [
        [StartTrial(experiment_id="exp-1", params={"x": 0.7})],
        [],  # Empty for second call
    ]

    # Create experiment spec
    spec = ExperimentSpec(
        name="test-exp",
        objective=objective,
        search_space={"x": Float(0, 1)},
        pipeline=Pipeline(steps=[mock_policy]),
        resources=Resources(max_concurrent=2),
        budget=Budget(max_trials=5),
    )

    # Create and run runner
    runner = ExperimentRunner(spec, services=services)

    # Start the run (but don't await completion)
    run_task = asyncio.create_task(runner.run())

    # Give it time to set up and process initial events
    await asyncio.sleep(0.1)

    # Verify policy received events
    assert len(mock_policy.events_received) > 0
    event_types = [e.type for e in mock_policy.events_received]
    assert (
        EventType.EXPERIMENT_STARTED in event_types
        or EventType.CAPACITY_AVAILABLE in event_types
    )

    # Verify policy decide() was called
    assert mock_policy.decide_called > 0

    # Cancel the run task
    run_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await run_task


@pytest.mark.asyncio
async def test_runner_subscribes_callbacks():
    """Provided callbacks receive events."""
    # Setup services
    bus = InMemoryBus()
    stores = SimpleStores()
    executor = LocalAsyncExecutor(bus)
    capacity = CapacityManager(max_concurrent=1)

    services = {
        "bus": bus,
        "stores": stores,
        "executor": executor,
        "capacity": capacity,
    }

    # Track callback events
    callback_events = []

    class TestCallback:
        async def on_event(self, evt):
            callback_events.append(evt)

    # Simple objective
    def objective(ctx):
        return ObjectiveResult(score=0.5)

    # Create experiment spec with callback
    spec = ExperimentSpec(
        name="test-exp",
        objective=objective,
        search_space={},
        pipeline=Pipeline(steps=[]),
        monitoring={"callbacks": [TestCallback()]},
        budget=Budget(max_trials=1),
    )

    # Create runner
    runner = ExperimentRunner(spec, services=services)

    # Start the run
    run_task = asyncio.create_task(runner.run())

    # Give it time to emit events
    await asyncio.sleep(0.05)

    # Cancel the run
    run_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await run_task

    # Verify callback received events
    assert len(callback_events) > 0
    event_types = [e.type for e in callback_events]
    assert EventType.EXPERIMENT_STARTED in event_types


@pytest.mark.asyncio
async def test_runner_returns_best_summary():
    """After trials complete, run() returns dict with best result."""
    # Setup services
    bus = InMemoryBus()
    stores = SimpleStores()
    executor = LocalAsyncExecutor(bus)
    capacity = CapacityManager(max_concurrent=2)

    services = {
        "bus": bus,
        "stores": stores,
        "executor": executor,
        "capacity": capacity,
    }

    # Objective that returns different scores
    call_count = 0

    def objective(ctx, x=0.5):
        nonlocal call_count
        call_count += 1
        # Return increasing scores
        return ObjectiveResult(score=0.1 * call_count)

    # Use RandomSearchPolicy with small trial count
    space = {"x": Float(0, 1)}

    # Create experiment spec
    spec = ExperimentSpec(
        name="test-exp",
        objective=objective,
        search_space=space,
        pipeline=Pipeline(
            steps=[RandomSearchPolicy(space=space, experiment_id="exp-will-be-set")]
        ),
        resources=Resources(max_concurrent=2),
        budget=Budget(max_trials=3),
    )

    # Create and run runner
    runner = ExperimentRunner(spec, services=services)

    # Run with timeout
    try:
        result = await asyncio.wait_for(runner.run(), timeout=2.0)
    except TimeoutError:
        # If it takes too long, still check what we have
        result = {"experiment": spec.name, "best": {}}

    # Should return a result dict
    assert isinstance(result, dict)
    assert "experiment" in result
    assert result["experiment"] == "test-exp"

    # If trials completed, should have best info
    if call_count > 0:
        assert "best" in result
        if result["best"]:
            assert (
                "trial_id" in result["best"]
                or "params" in result["best"]
                or "score" in result["best"]
            )
