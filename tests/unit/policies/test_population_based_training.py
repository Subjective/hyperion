"""Tests for Population-Based Training strategy."""

from typing import Literal

import pytest

from hyperion.core.events import Event, EventType
from hyperion.core.state import TrialView
from hyperion.framework.policy import StartTrial
from hyperion.framework.search_space import Bool, Choice, Float, Int
from hyperion.policies.population_based_training import PopulationBasedTrainingPolicy


class MockTrialView:
    """Mock trial view for testing."""

    def __init__(
        self,
        trial_id,
        status="RUNNING",
        score=None,
        params=None,
        metrics_last=None,
    ):
        self.trial_id = trial_id
        self.status = status
        self.score = score
        self.params = params or {"x": 0.5, "y": 32}
        self.metrics_last = metrics_last or {"loss": 0.5}
        self.experiment_id = "exp-1"
        self.parent_trial_id = None
        self.depth = 0
        self.branch_id = None
        self.tags = {}


class MockReadableState:
    """Mock implementation of ReadableState for testing."""

    def __init__(self, running_trials=None, completed_trials=None):
        self.running = running_trials or []
        self.completed = completed_trials or []

    def capacity_free(self) -> int:
        return max(0, 8 - len(self.running))

    def all_trials(self, experiment_id: str) -> list[TrialView]:
        return self.running + self.completed

    def completed_trials(self, experiment_id: str) -> list[TrialView]:
        return self.completed

    def best_trials(
        self,
        experiment_id: str,
        top_n: int,
        key: str = "score",
        mode: Literal["min", "max"] = "max",
    ) -> list[TrialView]:
        all_completed = [t for t in self.completed if t.score is not None]
        sorted_trials = sorted(
            all_completed, key=lambda t: t.score, reverse=(mode == "max")
        )
        return sorted_trials[:top_n]

    def trial(self, trial_id: str) -> TrialView | None:
        for t in self.running + self.completed:
            if t.trial_id == trial_id:
                return t
        return None

    def running_trials(self, experiment_id: str | None = None) -> list[TrialView]:
        return self.running

    def trials_by_depth(self, experiment_id: str, depth: int) -> list[TrialView]:
        return []


@pytest.mark.asyncio
async def test_pbt_bootstraps_population():
    """PBT should bootstrap initial population."""
    space = {"x": Float(0.0, 1.0), "y": Int(16, 128)}

    policy = PopulationBasedTrainingPolicy(
        space=space, population_size=4, experiment_id="exp-1"
    )

    state = MockReadableState(running_trials=[])
    actions = await policy.decide(state)

    # Should start population_size trials
    assert len(actions) == 4
    assert all(isinstance(a, StartTrial) for a in actions)

    # Check parameters are valid
    for action in actions:
        assert isinstance(action, StartTrial)
        assert 0.0 <= action.params["x"] <= 1.0
        assert 16 <= action.params["y"] <= 128
        assert action.tags is not None
        assert action.tags["strategy"] == "pbt"
        assert action.tags["pbt_generation"] == 0
        assert action.parent_trial_id is None  # Initial population has no parents


@pytest.mark.asyncio
async def test_pbt_maintains_population_size():
    """PBT should maintain population size during bootstrap."""
    policy = PopulationBasedTrainingPolicy(
        space={"x": Float(0, 1)}, population_size=4, experiment_id="exp-1"
    )

    # Population partially filled
    running = [MockTrialView(f"trial-{i}") for i in range(2)]
    state = MockReadableState(running_trials=running)
    actions = await policy.decide(state)

    # Should start 2 more to reach population size
    assert len(actions) == 2
    assert all(isinstance(a, StartTrial) for a in actions)


@pytest.mark.asyncio
async def test_pbt_evolves_from_completed_trials():
    """PBT should evolve new generation from completed trials."""
    policy = PopulationBasedTrainingPolicy(
        space={"x": Float(0, 1), "y": Int(1, 10)},
        population_size=4,
        experiment_id="exp-1",
    )

    # Simulate completed initial population with varying performance
    completed = [
        MockTrialView(f"trial-{i}", status="COMPLETED", score=0.1 * i) for i in range(4)
    ]
    state = MockReadableState(running_trials=[], completed_trials=completed)

    actions = await policy.decide(state)

    # Should start new generation from best performers
    assert len(actions) > 0
    assert all(isinstance(a, StartTrial) for a in actions)

    # New trials should have parent_trial_id set
    for action in actions:
        assert isinstance(action, StartTrial)
        assert action.parent_trial_id is not None
        # Parent should be one of the better performers
        parent_scores = [
            t.score for t in completed if t.trial_id == action.parent_trial_id
        ]
        assert len(parent_scores) > 0
        # Should be from top half
        assert parent_scores[0] is not None
        assert parent_scores[0] >= 0.2  # trial-2 or trial-3

        # Check tags
        assert action.tags is not None
        assert action.tags["strategy"] == "pbt"
        assert action.tags["pbt_generation"] == 1
        assert "pbt_parent" in action.tags


@pytest.mark.asyncio
async def test_pbt_no_experiment_id_returns_no_actions():
    """PBT should not propose actions without experiment_id."""
    policy = PopulationBasedTrainingPolicy(space={"x": Float(0, 1)}, experiment_id=None)

    state = MockReadableState()
    actions = await policy.decide(state)

    assert actions == []


@pytest.mark.asyncio
async def test_pbt_tracks_experiment_started():
    """PBT should track experiment started event to get experiment_id."""
    policy = PopulationBasedTrainingPolicy(space={"x": Float(0, 1)}, experiment_id=None)

    # Initially no experiment_id
    assert policy.experiment_id is None

    # Simulate experiment started event
    events = [
        Event(type=EventType.EXPERIMENT_STARTED, data={"experiment_id": "exp-123"})
    ]

    await policy.on_events(events)

    # Should now have experiment_id
    assert policy.experiment_id == "exp-123"


@pytest.mark.asyncio
async def test_pbt_perturbs_parameters():
    """PBT should correctly perturb different parameter types."""
    space = {
        "continuous": Float(0.0, 1.0),
        "integer": Int(1, 100),
        "categorical": Choice(["a", "b", "c"]),
        "boolean": Bool(),
    }

    policy = PopulationBasedTrainingPolicy(
        space=space, perturbation_factor=1.2, experiment_id="exp-1"
    )

    original_params = {
        "continuous": 0.5,
        "integer": 50,
        "categorical": "b",
        "boolean": True,
    }

    # Test perturbation multiple times
    for _ in range(10):
        perturbed = policy._perturb_params(original_params)

        # Continuous should be modified but within bounds
        assert 0.0 <= perturbed["continuous"] <= 1.0

        # Integer should be modified but within bounds
        assert 1 <= perturbed["integer"] <= 100

        # Categorical should be from options
        assert perturbed["categorical"] in ["a", "b", "c"]

        # Boolean should be boolean
        assert isinstance(perturbed["boolean"], bool)


@pytest.mark.asyncio
async def test_pbt_creates_lineage():
    """PBT should create proper trial lineage through parent_trial_id."""
    policy = PopulationBasedTrainingPolicy(
        space={"x": Float(0, 1)},
        population_size=4,
        experiment_id="exp-1",
    )

    # Generation 0: Initial population
    state = MockReadableState()
    actions = await policy.decide(state)

    # Initial trials have no parents
    for action in actions:
        assert isinstance(action, StartTrial)
        assert action.parent_trial_id is None
        assert action.tags is not None
        assert action.tags["pbt_generation"] == 0

    # Generation 1: Evolve from completed
    completed = [
        MockTrialView("trial-best", status="COMPLETED", score=0.9),
        MockTrialView("trial-good", status="COMPLETED", score=0.7),
        MockTrialView("trial-ok", status="COMPLETED", score=0.5),
        MockTrialView("trial-bad", status="COMPLETED", score=0.3),
    ]
    state = MockReadableState(completed_trials=completed)

    # Update last evolution count to trigger evolution
    policy.last_evolution_count = 0
    actions = await policy.decide(state)

    # New generation should have parents
    assert len(actions) > 0
    for action in actions:
        assert isinstance(action, StartTrial)
        assert action.parent_trial_id in ["trial-best", "trial-good"]
        assert action.tags is not None
        assert action.tags["pbt_generation"] == 1
        assert action.tags["pbt_parent"] == action.parent_trial_id


@pytest.mark.asyncio
async def test_pbt_waits_for_enough_completed():
    """PBT should wait for enough trials to complete before evolving."""
    policy = PopulationBasedTrainingPolicy(
        space={"x": Float(0, 1)},
        population_size=8,
        experiment_id="exp-1",
    )

    # Not enough completed trials (need at least population_size // 2 = 4)
    completed = [
        MockTrialView(f"trial-{i}", status="COMPLETED", score=0.5) for i in range(3)
    ]
    running = [MockTrialView(f"trial-r-{i}") for i in range(5)]
    state = MockReadableState(running_trials=running, completed_trials=completed)

    actions = await policy.decide(state)

    # Should not evolve yet - not enough completed
    assert len(actions) == 0

    # Now with enough completed
    completed = [
        MockTrialView(f"trial-{i}", status="COMPLETED", score=0.5) for i in range(4)
    ]
    state = MockReadableState(running_trials=running, completed_trials=completed)

    actions = await policy.decide(state)

    # Should evolve now
    if state.capacity_free() > 0:
        assert len(actions) > 0
        for action in actions:
            assert isinstance(action, StartTrial)
            assert action.parent_trial_id is not None


@pytest.mark.asyncio
async def test_pbt_rationale():
    """PBT should provide informative rationale."""
    policy = PopulationBasedTrainingPolicy(
        space={"x": Float(0, 1)},
        population_size=8,
        perturbation_factor=1.5,
        experiment_id="exp-1",
    )

    rationale = await policy.rationale()
    assert rationale is not None
    assert "Population-Based Training" in rationale
    assert "Generation 0" in rationale
    assert "population size 8" in rationale
    assert "perturbation factor 1.5" in rationale
