"""Tests for emitting EXPERIMENT_COMPLETED and updating status."""

import asyncio

import pytest

from hyperion.core.bus import InMemoryBus
from hyperion.core.capacity import CapacityManager
from hyperion.core.events import EventType
from hyperion.core.executor import LocalAsyncExecutor
from hyperion.core.models import ExperimentStatus, ObjectiveResult
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
async def test_experiment_completed_event_and_status():
    # Services
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

    # Simple objective
    def obj(ctx, x=0.1):
        return ObjectiveResult(score=float(x))

    space = {"x": Float(0, 1)}

    spec = ExperimentSpec(
        name="exp-completion",
        objective=obj,
        search_space=space,
        pipeline=Pipeline(steps=[RandomSearchPolicy(space=space)]),
        resources=Resources(max_concurrent=1),
        budget=Budget(max_trials=1),
    )

    runner = ExperimentRunner(spec, services=services)
    result = await asyncio.wait_for(runner.run(), timeout=2.0)

    # Verify result structure
    assert result["experiment"] == "exp-completion"

    # Verify EXPERIMENT_COMPLETED was emitted
    events = await stores.events.tail(100)
    types = [e.type for e in events]
    assert EventType.EXPERIMENT_COMPLETED in types

    # Verify experiment status is COMPLETED
    assert runner.experiment_id is not None
    exp = stores.experiments.get(runner.experiment_id)
    assert exp is not None
    assert exp.status == ExperimentStatus.COMPLETED
