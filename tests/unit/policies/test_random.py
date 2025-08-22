"""Tests for RandomSearchPolicy strategy."""

from typing import Literal

import pytest

from hyperion.core.events import Event, EventType
from hyperion.core.state import TrialView
from hyperion.framework.policy import StartTrial
from hyperion.framework.search_space import Choice, Float
from hyperion.policies.random_search import RandomSearchPolicy


class MockReadableState:
    """Mock implementation of ReadableState for testing."""

    def __init__(self, running_count=0, completed_count=0):
        self.running_count = running_count
        self.completed_count = completed_count
        self.total_trials = running_count + completed_count

    def capacity_free(self) -> int:
        return 4 - self.running_count  # Assume max 4 concurrent

    def trial(self, trial_id: str) -> TrialView | None:
        return None

    def running_trials(self, experiment_id: str | None = None) -> list[TrialView]:
        return [
            TrialView(
                trial_id=f"trial-{i}",
                experiment_id="exp-1",
                params={},
                status="RUNNING",
                score=None,
                metrics_last={},
                parent_trial_id=None,
                depth=0,
                branch_id=None,
                tags={},
            )
            for i in range(self.running_count)
        ]

    def best_trials(
        self,
        experiment_id: str,
        top_n: int,
        key: str = "score",
        mode: Literal["min", "max"] = "max",
    ) -> list[TrialView]:
        return []

    def trials_by_depth(self, experiment_id: str, depth: int) -> list[TrialView]:
        return []

    def completed_trials(self, experiment_id: str) -> list[TrialView]:
        """Get all completed trials for an experiment."""
        return []

    def all_trials(self, experiment_id: str) -> list[TrialView]:
        """Get all trials regardless of status."""
        return self.running_trials(experiment_id)


@pytest.mark.asyncio
async def test_random_policy_samples_from_space():
    """RandomSearchPolicy should sample parameters from search space."""
    space = {"lr": Float(0.001, 0.1), "optimizer": Choice(["sgd", "adam"])}

    policy = RandomSearchPolicy(space=space, experiment_id="exp-1")

    state = MockReadableState(running_count=0, completed_count=0)
    actions = await policy.decide(state)

    # Should produce StartTrial actions
    assert len(actions) > 0
    assert all(isinstance(a, StartTrial) for a in actions)

    # Parameters should be from search space
    for action in actions:
        assert isinstance(action, StartTrial)
        assert 0.001 <= action.params["lr"] <= 0.1
        assert action.params["optimizer"] in ["sgd", "adam"]
        assert action.experiment_id == "exp-1"


@pytest.mark.asyncio
async def test_random_policy_no_experiment_id_returns_no_actions():
    """RandomSearchPolicy should not propose actions until experiment_id is set."""
    policy = RandomSearchPolicy(space={"x": Float(0, 1)}, experiment_id=None)
    state = MockReadableState(running_count=0, completed_count=0)
    actions = await policy.decide(state)
    assert actions == []


@pytest.mark.asyncio
async def test_random_policy_respects_capacity():
    """RandomSearchPolicy should only start trials if capacity available."""
    policy = RandomSearchPolicy(space={"x": Float(0, 1)}, experiment_id="exp-1")

    # State with 4 running trials (no capacity)
    state = MockReadableState(running_count=4, completed_count=0)
    actions = await policy.decide(state)

    # Should not start more trials when at capacity
    assert len(actions) == 0

    # State with 2 running trials (2 slots free)
    state = MockReadableState(running_count=2, completed_count=0)
    actions = await policy.decide(state)

    # Should start up to 2 trials
    assert 0 < len(actions) <= 2


@pytest.mark.asyncio
async def test_random_policy_tracks_trial_count():
    """RandomSearchPolicy should track started and completed trials."""
    policy = RandomSearchPolicy(space={"x": Float(0, 1)}, experiment_id="exp-1")

    # Start some trials
    state = MockReadableState(running_count=0, completed_count=0)
    actions = await policy.decide(state)
    initial_count = len(actions)

    # Simulate completions
    await policy.on_events(
        [
            Event(type=EventType.TRIAL_COMPLETED, data={"trial_id": f"t{i}"})
            for i in range(initial_count)
        ]
    )

    # Should be able to start more trials (up to max_trials)
    state = MockReadableState(running_count=0, completed_count=initial_count)
    actions = await policy.decide(state)

    # Should start more trials based on capacity
    assert len(actions) > 0


@pytest.mark.asyncio
async def test_random_policy_rationale():
    """RandomSearchPolicy should provide a rationale."""
    policy = RandomSearchPolicy(space={"x": Float(0, 1)}, experiment_id="exp-1")

    rationale = await policy.rationale()
    assert rationale is not None
    assert "random" in rationale.lower()
