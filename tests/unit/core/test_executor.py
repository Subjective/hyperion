"""Tests for trial executor implementations."""

import asyncio
import time

import pytest

from hyperion.core.bus import InMemoryBus
from hyperion.core.context import TrialContext
from hyperion.core.events import EventType
from hyperion.core.executor import LocalAsyncExecutor, LocalProcessExecutor
from hyperion.core.models import ObjectiveResult


@pytest.mark.parametrize("executor_cls", [LocalAsyncExecutor, LocalProcessExecutor])
@pytest.mark.asyncio
async def test_submit_emits_started_and_completed_events(executor_cls):
    """Submitting a trivial objective should emit TRIAL_STARTED then TRIAL_COMPLETED with score."""
    bus = InMemoryBus()
    executor = executor_cls(bus)
    events = []

    bus.subscribe("*", lambda e: events.append(e))

    # Trivial objective that returns immediately
    def objective(ctx: TrialContext, x: float) -> ObjectiveResult:
        return ObjectiveResult(score=x * 2, metrics={"x": x})

    # For process executor, submission returns immediately; await completion via drain
    asyncio.create_task(executor.submit("trial-1", "exp-1", objective, {"x": 0.5}, {}))

    # Wait until completed is observed
    async def _wait_completed():
        for _ in range(200):
            if any(e.type == EventType.TRIAL_COMPLETED for e in events):
                return
            await asyncio.sleep(0.01)
        raise AssertionError("timed out waiting for TRIAL_COMPLETED")

    await _wait_completed()
    await bus.drain()

    # Should have TRIAL_STARTED and TRIAL_COMPLETED events
    event_types = [e.type for e in events]
    assert EventType.TRIAL_STARTED in event_types
    assert EventType.TRIAL_COMPLETED in event_types

    # Find completed event and check score
    completed = [e for e in events if e.type == EventType.TRIAL_COMPLETED][0]
    assert completed.data["trial_id"] == "trial-1"
    assert completed.data["score"] == 1.0
    assert completed.data["metrics"]["x"] == 0.5


@pytest.mark.parametrize("executor_cls", [LocalAsyncExecutor, LocalProcessExecutor])
@pytest.mark.asyncio
async def test_kill_sets_should_stop_and_emits_killed(executor_cls):
    """Objective that loops on ctx.should_stop() stops after kill(), emitting TRIAL_KILLED."""
    bus = InMemoryBus()
    executor = executor_cls(bus)
    events = []

    bus.subscribe("*", lambda e: events.append(e))

    # Objective that loops until should_stop
    def objective(ctx: TrialContext) -> ObjectiveResult:
        while not ctx.should_stop():
            # In thread context, we can't use asyncio.sleep
            time.sleep(0.01)
        return ObjectiveResult(score=0.0)

    # Start the trial
    asyncio.create_task(executor.submit("trial-1", "exp-1", objective, {}, {}))

    # Wait a bit for it to start
    await asyncio.sleep(0.05)

    # Kill it
    await executor.kill("trial-1")

    # Wait for task to complete
    # Wait until killed is observed (process-based may take a moment)
    async def _wait_killed():
        for _ in range(200):
            if any(e.type == EventType.TRIAL_KILLED for e in events):
                return
            await asyncio.sleep(0.01)
        raise AssertionError("timed out waiting for TRIAL_KILLED")

    await _wait_killed()
    await bus.drain()

    # Should have TRIAL_KILLED event
    event_types = [e.type for e in events]
    assert EventType.TRIAL_KILLED in event_types

    killed = [e for e in events if e.type == EventType.TRIAL_KILLED][0]
    assert killed.data["trial_id"] == "trial-1"


@pytest.mark.parametrize("executor_cls", [LocalAsyncExecutor, LocalProcessExecutor])
@pytest.mark.asyncio
async def test_exception_in_objective_emits_failed(executor_cls):
    """Objective raising an exception leads to TRIAL_FAILED with error info."""
    bus = InMemoryBus()
    executor = executor_cls(bus)
    events = []

    bus.subscribe("*", lambda e: events.append(e))

    # Objective that raises an error
    def objective(ctx: TrialContext) -> ObjectiveResult:
        raise ValueError("Test error")

    asyncio.create_task(executor.submit("trial-1", "exp-1", objective, {}, {}))

    # Wait until failed is observed
    async def _wait_failed():
        for _ in range(200):
            if any(e.type == EventType.TRIAL_FAILED for e in events):
                return
            await asyncio.sleep(0.01)
        raise AssertionError("timed out waiting for TRIAL_FAILED")

    await _wait_failed()
    await bus.drain()

    # Should have TRIAL_FAILED event
    event_types = [e.type for e in events]
    assert EventType.TRIAL_FAILED in event_types

    failed = [e for e in events if e.type == EventType.TRIAL_FAILED][0]
    assert failed.data["trial_id"] == "trial-1"
    assert "ValueError" in failed.data["error"]
    assert "Test error" in failed.data["error"]


@pytest.mark.parametrize("executor_cls", [LocalAsyncExecutor, LocalProcessExecutor])
@pytest.mark.asyncio
async def test_progress_reports_emit_trial_progress(executor_cls):
    """Objective calling ctx.report(step, **metrics) results in TRIAL_PROGRESS events."""
    bus = InMemoryBus()
    executor = executor_cls(bus)
    events = []

    bus.subscribe("*", lambda e: events.append(e))

    # Objective that reports progress
    def objective(ctx: TrialContext) -> ObjectiveResult:
        ctx.report(0, loss=1.0, accuracy=0.1)
        ctx.report(1, loss=0.5, accuracy=0.5)
        ctx.report(2, loss=0.1, accuracy=0.9)
        return ObjectiveResult(score=0.9)

    asyncio.create_task(executor.submit("trial-1", "exp-1", objective, {}, {}))

    # Wait until 3 progress events are observed
    async def _wait_progress():
        for _ in range(200):
            if len([e for e in events if e.type == EventType.TRIAL_PROGRESS]) >= 3:
                return
            await asyncio.sleep(0.01)
        raise AssertionError("timed out waiting for TRIAL_PROGRESS events")

    await _wait_progress()
    await bus.drain()

    # Should have TRIAL_PROGRESS events
    progress_events = [e for e in events if e.type == EventType.TRIAL_PROGRESS]
    assert len(progress_events) == 3

    # Check first progress event
    assert progress_events[0].data["trial_id"] == "trial-1"
    assert progress_events[0].data["step"] == 0
    assert progress_events[0].data["metrics"]["loss"] == 1.0
    assert progress_events[0].data["metrics"]["accuracy"] == 0.1

    # Check last progress event
    assert progress_events[2].data["step"] == 2
    assert progress_events[2].data["metrics"]["loss"] == 0.1
    assert progress_events[2].data["metrics"]["accuracy"] == 0.9


@pytest.mark.parametrize("executor_cls", [LocalAsyncExecutor, LocalProcessExecutor])
@pytest.mark.asyncio
async def test_events_use_experiment_id_as_aggregate_id(executor_cls):
    """All emitted events should use experiment_id as aggregate_id."""
    bus = InMemoryBus()
    executor = executor_cls(bus)
    events = []

    bus.subscribe("*", lambda e: events.append(e))

    # Simple objective
    def objective(ctx: TrialContext) -> ObjectiveResult:
        ctx.report(0, loss=1.0)
        return ObjectiveResult(score=0.5)

    experiment_id = "test-exp-123"
    trial_id = "test-trial-456"

    asyncio.create_task(executor.submit(trial_id, experiment_id, objective, {}, {}))

    # Wait for completion
    async def _wait_completed():
        for _ in range(200):
            if any(e.type == EventType.TRIAL_COMPLETED for e in events):
                return
            await asyncio.sleep(0.01)
        raise AssertionError("timed out waiting for TRIAL_COMPLETED")

    await _wait_completed()
    await bus.drain()

    # All events should have experiment_id as aggregate_id
    for event in events:
        assert event.aggregate_id == experiment_id, (
            f"Event {event.type} has wrong aggregate_id"
        )
