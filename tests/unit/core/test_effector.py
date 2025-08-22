"""Tests for the Effector command emitter."""

import pytest

from hyperion.core.bus import InMemoryBus
from hyperion.core.effector import Effector
from hyperion.core.events import CommandType


@pytest.mark.asyncio
async def test_effector_start_trial():
    """Effector should publish START_TRIAL commands."""
    bus = InMemoryBus()
    published = []

    bus.subscribe(CommandType.START_TRIAL, lambda cmd: published.append(cmd))

    effector = Effector(bus)
    await effector.start_trial(
        experiment_id="exp-1",
        params={"lr": 0.01},
        parent_trial_id="parent-123",
        tags={"type": "exploration"},
    )
    await bus.drain()

    assert len(published) == 1
    cmd = published[0]
    assert cmd.type == CommandType.START_TRIAL
    assert cmd.data["experiment_id"] == "exp-1"
    assert cmd.data["params"]["lr"] == 0.01
    assert cmd.data["parent_trial_id"] == "parent-123"
    assert cmd.data["tags"]["type"] == "exploration"


@pytest.mark.asyncio
async def test_effector_kill_trial():
    """Effector should publish KILL_TRIAL commands."""
    bus = InMemoryBus()
    published = []

    bus.subscribe(CommandType.KILL_TRIAL, lambda cmd: published.append(cmd))

    effector = Effector(bus)
    await effector.kill_trial("trial-456", reason="Early stopping")
    await bus.drain()

    assert len(published) == 1
    cmd = published[0]
    assert cmd.type == CommandType.KILL_TRIAL
    assert cmd.data["trial_id"] == "trial-456"
    assert cmd.data["reason"] == "Early stopping"


@pytest.mark.asyncio
async def test_effector_patch_trial():
    """Effector should publish PATCH_TRIAL commands."""
    bus = InMemoryBus()
    published = []

    bus.subscribe(CommandType.PATCH_TRIAL, lambda cmd: published.append(cmd))

    effector = Effector(bus)
    await effector.patch_trial("trial-789", {"lr": 0.001, "batch_size": 64})
    await bus.drain()

    assert len(published) == 1
    cmd = published[0]
    assert cmd.type == CommandType.PATCH_TRIAL
    assert cmd.data["trial_id"] == "trial-789"
    assert cmd.data["patch"]["lr"] == 0.001
    assert cmd.data["patch"]["batch_size"] == 64


@pytest.mark.asyncio
async def test_effector_start_experiment():
    """Effector should publish START_EXPERIMENT commands."""
    bus = InMemoryBus()
    published = []

    bus.subscribe(CommandType.START_EXPERIMENT, lambda cmd: published.append(cmd))

    effector = Effector(bus)
    await effector.start_experiment({"name": "test-exp", "config": {"max_trials": 100}})
    await bus.drain()

    assert len(published) == 1
    cmd = published[0]
    assert cmd.type == CommandType.START_EXPERIMENT
    assert cmd.data["name"] == "test-exp"
    assert cmd.data["config"]["max_trials"] == 100
