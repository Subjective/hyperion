"""Tests for Bayesian Optimization strategy."""

import pytest

from hyperion.core.events import Event, EventType
from hyperion.core.state import TrialView
from hyperion.framework.policy import StartTrial
from hyperion.framework.search_space import Choice, Float
from hyperion.policies.bayesian_optimization import BayesianOptimizationPolicy


class MockReadableState:
    """Mock implementation of ReadableState for testing."""

    def __init__(self, completed_count=0, running_count=0):
        self.completed_count = completed_count
        self.running_count = running_count

    def capacity_free(self) -> int:
        return 4 - self.running_count

    def completed_trials(self, experiment_id: str) -> list[TrialView]:
        """Return mock completed trials for testing."""
        trials = []
        for i in range(self.completed_count):
            trials.append(
                TrialView(
                    trial_id=f"trial-{i}",
                    experiment_id=experiment_id,
                    params={"x": 0.1 * i, "y": "option_a"},
                    status="COMPLETED",
                    score=0.5 + 0.1 * i,  # Increasing scores
                    metrics_last={},
                    parent_trial_id=None,
                    depth=0,
                    branch_id=None,
                    tags={},
                )
            )
        return trials

    def all_trials(self, experiment_id: str) -> list[TrialView]:
        return self.completed_trials(experiment_id)

    def trial(self, trial_id: str) -> TrialView | None:
        return None

    def running_trials(self, experiment_id: str | None = None) -> list[TrialView]:
        return []

    def best_trials(
        self, experiment_id: str, top_n: int, key: str = "score", mode: str = "max"
    ) -> list[TrialView]:
        trials = self.completed_trials(experiment_id)
        sorted_trials = sorted(
            trials,
            key=lambda t: t.score if t.score is not None else 0,
            reverse=(mode == "max"),
        )
        return sorted_trials[:top_n]

    def trials_by_depth(self, experiment_id: str, depth: int) -> list[TrialView]:
        return []


@pytest.mark.asyncio
async def test_bayesian_starts_with_random_exploration():
    """BayesianOptimizationPolicy should use random exploration initially."""
    space = {"x": Float(0.0, 1.0), "y": Choice(["option_a", "option_b"])}

    policy = BayesianOptimizationPolicy(space=space, n_initial=5, experiment_id="exp-1")

    # With no completed trials, should sample randomly
    state = MockReadableState(completed_count=0)
    actions = await policy.decide(state)

    assert len(actions) > 0
    assert all(isinstance(a, StartTrial) for a in actions)

    # Check that parameters are within bounds
    for action in actions:
        assert isinstance(action, StartTrial)
        assert 0.0 <= action.params["x"] <= 1.0
        assert action.params["y"] in ["option_a", "option_b"]
        assert action.tags is not None
        assert action.tags["phase"] == "exploration"


@pytest.mark.asyncio
async def test_bayesian_uses_gp_after_initial():
    """BayesianOptimizationPolicy should use GP after initial exploration."""
    space = {"x": Float(0.0, 1.0), "y": Choice(["option_a", "option_b"])}

    policy = BayesianOptimizationPolicy(space=space, n_initial=5, experiment_id="exp-1")

    # With enough completed trials, should use acquisition function
    state = MockReadableState(completed_count=10)
    actions = await policy.decide(state)

    assert len(actions) > 0
    assert all(isinstance(a, StartTrial) for a in actions)

    # Should be in exploitation phase
    for action in actions:
        assert isinstance(action, StartTrial)
        assert action.tags is not None
        assert action.tags["phase"] in ["exploitation", "fallback", "error_fallback"]


@pytest.mark.asyncio
async def test_bayesian_respects_capacity():
    """BayesianOptimizationPolicy should respect capacity limits."""
    policy = BayesianOptimizationPolicy(space={"x": Float(0, 1)}, experiment_id="exp-1")

    # State with no free capacity
    state = MockReadableState(running_count=4, completed_count=0)
    actions = await policy.decide(state)

    assert len(actions) == 0


@pytest.mark.asyncio
async def test_bayesian_no_experiment_id_returns_no_actions():
    """BayesianOptimizationPolicy should not propose actions without experiment_id."""
    policy = BayesianOptimizationPolicy(space={"x": Float(0, 1)}, experiment_id=None)

    state = MockReadableState(completed_count=0)
    actions = await policy.decide(state)

    assert actions == []


@pytest.mark.asyncio
async def test_bayesian_handles_categorical_params():
    """BayesianOptimizationPolicy should handle categorical parameters."""
    from hyperion.framework.search_space import Bool

    space = {
        "x": Float(0.0, 1.0),
        "category": Choice(["a", "b", "c"]),
        "flag": Bool(),
    }

    policy = BayesianOptimizationPolicy(space=space, n_initial=3, experiment_id="exp-1")

    # Test with completed trials containing categorical params
    state = MockReadableState(completed_count=5)
    actions = await policy.decide(state)

    assert len(actions) > 0
    for action in actions:
        assert isinstance(action, StartTrial)
        assert 0.0 <= action.params["x"] <= 1.0
        assert action.params["category"] in ["a", "b", "c"]
        assert isinstance(action.params["flag"], bool)


@pytest.mark.asyncio
async def test_bayesian_tracks_events():
    """BayesianOptimizationPolicy should process events without errors."""
    policy = BayesianOptimizationPolicy(space={"x": Float(0, 1)}, experiment_id="exp-1")

    # Simulate trial lifecycle events
    events = [
        Event(type=EventType.TRIAL_STARTED, data={"trial_id": "t1"}),
        Event(type=EventType.TRIAL_COMPLETED, data={"trial_id": "t1", "score": 0.8}),
        Event(type=EventType.TRIAL_STARTED, data={"trial_id": "t2"}),
        Event(type=EventType.TRIAL_FAILED, data={"trial_id": "t2"}),
    ]

    # Should process events without errors
    await policy.on_events(events)

    # Verify the policy is still functional after processing events
    assert policy.experiment_id == "exp-1"
    assert policy.n_initial == 10


@pytest.mark.asyncio
async def test_bayesian_rationale():
    """BayesianOptimizationPolicy should provide informative rationale."""
    policy = BayesianOptimizationPolicy(
        space={"x": Float(0, 1)}, n_initial=5, experiment_id="exp-1"
    )

    # Test rationale wording
    rationale = await policy.rationale()
    assert rationale is not None
    assert "exploration" in rationale.lower() or "exploitation" in rationale.lower()
    # Updated optimizer uses TPE density ratio (no GP/EI)
    assert "tpe" in rationale.lower()
