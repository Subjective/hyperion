"""Tests for event metadata (aggregate_id, correlation_id, causation_id)."""

import asyncio

import pytest

from hyperion.core.bus import InMemoryBus
from hyperion.core.capacity import CapacityManager
from hyperion.core.events import CommandType, EventType
from hyperion.core.executor import LocalAsyncExecutor
from hyperion.core.models import ObjectiveResult
from hyperion.framework.experiment import (
    Budget,
    ExperimentRunner,
    ExperimentSpec,
    Pipeline,
    Resources,
)
from hyperion.framework.search_space import Float
from hyperion.policies.random_search import RandomSearchPolicy
from hyperion.storage.memory import (
    InMemoryEventLog,
    InMemoryExperimentStore,
    InMemoryTrialStore,
)


class SimpleStores:
    def __init__(self):
        self.events = InMemoryEventLog()
        self.trials = InMemoryTrialStore()
        self.experiments = InMemoryExperimentStore()


@pytest.mark.asyncio
async def test_event_metadata_chain_and_capacity_aggregate():
    bus = InMemoryBus()
    stores = SimpleStores()
    exec_impl = LocalAsyncExecutor(bus)
    capacity = CapacityManager(max_concurrent=1)

    services = {
        "bus": bus,
        "stores": stores,
        "executor": exec_impl,
        "capacity": capacity,
    }

    def objective(ctx, x=0.5):
        return ObjectiveResult(score=float(x))

    space = {"x": Float(0, 1)}
    spec = ExperimentSpec(
        name="meta-test",
        objective=objective,
        search_space=space,
        pipeline=Pipeline(steps=[RandomSearchPolicy(space=space)]),
        resources=Resources(max_concurrent=1),
        budget=Budget(max_trials=2),
    )

    runner = ExperimentRunner(spec, services=services)
    await asyncio.wait_for(runner.run(), timeout=3.0)

    events = await stores.events.tail(500)

    # Gather ids by type
    start_exp_cmd_id = None
    start_trial_cmd_ids: list[str] = []
    trial_started_ids: list[str] = []
    terminal_ids: list[str] = []
    capacity_events = []
    exp_started = None
    exp_completed = None

    for e in events:
        if e.type == CommandType.START_EXPERIMENT:
            start_exp_cmd_id = e.id
        elif e.type == EventType.EXPERIMENT_STARTED:
            exp_started = e
        elif e.type == CommandType.START_TRIAL:
            start_trial_cmd_ids.append(e.id)
        elif e.type == EventType.TRIAL_STARTED:
            trial_started_ids.append(e.id)
        elif e.type in (
            EventType.TRIAL_COMPLETED,
            EventType.TRIAL_FAILED,
            EventType.TRIAL_KILLED,
        ):
            terminal_ids.append(e.id)
        elif e.type == EventType.CAPACITY_AVAILABLE:
            capacity_events.append(e)
        elif e.type == EventType.EXPERIMENT_COMPLETED:
            exp_completed = e

    # Basic expectations
    assert start_exp_cmd_id is not None
    assert exp_started is not None
    assert exp_completed is not None
    assert len(start_trial_cmd_ids) >= 1
    assert len(trial_started_ids) >= 1
    assert len(terminal_ids) >= 1
    assert len(capacity_events) >= 1

    # EXPERIMENT_STARTED should be caused by START_EXPERIMENT and share correlation
    assert exp_started.causation_id == start_exp_cmd_id
    # MAY be empty correlation on exp_started; not required here

    # TRIAL_* should correlate to their originating START_TRIAL
    start_trial_cmd_set = set(start_trial_cmd_ids)
    for e in events:
        if e.type in (
            EventType.TRIAL_STARTED,
            EventType.TRIAL_COMPLETED,
            EventType.TRIAL_FAILED,
            EventType.TRIAL_KILLED,
        ):
            assert e.correlation_id in start_trial_cmd_set
            assert e.aggregate_id == runner.experiment_id

    # CAPACITY_AVAILABLE should have aggregate_id == experiment and correlation inherited
    for cap in capacity_events:
        assert cap.aggregate_id == runner.experiment_id
        assert cap.causation_id in set(terminal_ids)

    # EXPERIMENT_COMPLETED should correlate to START_EXPERIMENT and (optionally) be caused by last terminal
    assert exp_completed.correlation_id == start_exp_cmd_id
    if terminal_ids:
        assert (
            exp_completed.causation_id in set(terminal_ids)
            or exp_completed.causation_id is None
        )
