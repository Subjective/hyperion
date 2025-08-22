"""Unit tests for LLM branching agent."""

import json
from unittest.mock import MagicMock

import pytest

from hyperion.core.events import Event, EventType
from hyperion.framework.policy import StartTrial
from hyperion.framework.search_space import Choice, Float, Int
from hyperion.policies.llm_branching_agent import (
    BranchingDecision,
    LLMBranchingAgent,
    TrialNode,
)


@pytest.fixture
def search_space():
    """Create a test search space."""
    return {
        "lr": Float(0.001, 0.1, log=True),
        "batch_size": Choice([16, 32, 64, 128]),
        "depth": Int(1, 5),
    }


@pytest.fixture
def agent(search_space):
    """Create an LLM branching agent instance."""

    def fake_llm(prompt: str) -> str:  # pragma: no cover - trivial stub
        return json.dumps({"decisions": [], "overall_strategy": "bootstrap"})

    return LLMBranchingAgent(
        space=search_space,
        experiment_id="test_exp",
        llm=fake_llm,
        max_depth=3,
        beam_width=2,
        branch_factor=2,
        enable_pruning=False,
    )


class TestTrialTreeManagement:
    """Test trial tree construction and management."""

    @pytest.mark.asyncio
    async def test_trial_tree_construction(self, agent):
        """Test that trial tree is built correctly from events."""
        # Simulate trial events
        events = [
            Event(
                type=EventType.TRIAL_STARTED,
                data={
                    "trial_id": "t1",
                    "params": {"lr": 0.01, "batch_size": 32},
                    "parent_trial_id": None,
                },
            ),
            Event(
                type=EventType.TRIAL_COMPLETED,
                data={"trial_id": "t1", "score": 0.85, "metrics": {"loss": 0.15}},
            ),
            Event(
                type=EventType.TRIAL_STARTED,
                data={
                    "trial_id": "t2",
                    "params": {"lr": 0.02, "batch_size": 64},
                    "parent_trial_id": "t1",
                },
            ),
        ]

        await agent.on_events(events)

        # Check tree structure
        assert "t1" in agent.trial_tree
        assert "t2" in agent.trial_tree

        t1_node = agent.trial_tree["t1"]
        assert t1_node.depth == 0
        assert t1_node.parent_id is None
        assert "t2" in t1_node.children
        assert t1_node.score == 0.85

        t2_node = agent.trial_tree["t2"]
        assert t2_node.depth == 1
        assert t2_node.parent_id == "t1"

    @pytest.mark.asyncio
    async def test_frontier_update(self, agent):
        """Test that frontier is updated correctly."""
        # Create completed trials at different depths
        agent.trial_tree = {
            "t1": TrialNode(
                trial_id="t1",
                params={"lr": 0.01},
                score=0.8,
                metrics={},
                status="COMPLETED",
                depth=0,
            ),
            "t2": TrialNode(
                trial_id="t2",
                params={"lr": 0.02},
                score=0.85,
                metrics={},
                status="COMPLETED",
                depth=0,
            ),
            "t3": TrialNode(
                trial_id="t3",
                params={"lr": 0.03},
                score=0.75,
                metrics={},
                status="COMPLETED",
                depth=0,
            ),
        }
        agent.trials_by_depth[0] = ["t1", "t2", "t3"]

        agent._update_frontier()

        # With beam_width=2, should keep top 2
        assert len(agent.frontier[0]) == 2
        assert "t2" in agent.frontier[0]  # Best score
        assert "t1" in agent.frontier[0]  # Second best


class TestBranchingDecisions:
    """Test LLM-based branching decision logic."""

    @pytest.mark.asyncio
    async def test_bootstrap_when_no_trials(self, agent):
        """Test bootstrap behavior when no trials exist."""
        state = MagicMock()
        state.capacity_free.return_value = 3

        actions = await agent.decide(state)

        # Should create bootstrap trials
        assert len(actions) > 0
        assert all(isinstance(a, StartTrial) for a in actions)
        assert all(a.parent_trial_id is None for a in actions)

    @pytest.mark.asyncio
    async def test_llm_branching_decision(self, agent):
        """Test LLM-based branching with mocked response."""
        # Setup trial tree
        agent.trial_tree = {
            "t1": TrialNode(
                trial_id="t1",
                params={"lr": 0.01, "batch_size": 32, "depth": 2},
                score=0.85,
                metrics={},
                status="COMPLETED",
                depth=0,
            ),
        }
        agent.trials_by_depth[0] = ["t1"]
        agent.frontier[0] = ["t1"]

        # Override agent.llm to a deterministic decision JSON
        agent.llm = lambda prompt: json.dumps(
            {
                "decisions": [
                    {
                        "parent_trial_id": "t1",
                        "rationale": "Trial t1 shows promise, exploring lr variations",
                        "parameter_focus": ["lr"],
                        "variations": [
                            {"lr": 0.008, "batch_size": 32, "depth": 2},
                            {"lr": 0.012, "batch_size": 32, "depth": 2},
                        ],
                        "confidence": "high",
                    }
                ],
                "overall_strategy": "Refining learning rate around best value",
            }
        )

        state = MagicMock()
        state.capacity_free.return_value = 2
        state.running_trials.return_value = []

        actions = await agent.decide(state)

        # Should create branches from t1
        assert len(actions) == 2
        assert all(isinstance(a, StartTrial) for a in actions)
        assert all(a.parent_trial_id == "t1" for a in actions)

        # Ensure actions were created as expected

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self, agent):
        """Test fallback behavior when LLM fails."""
        # Setup trial tree
        agent.trial_tree = {
            "t1": TrialNode(
                trial_id="t1",
                params={"lr": 0.01, "batch_size": 32, "depth": 2},
                score=0.85,
                metrics={},
                status="COMPLETED",
                depth=0,
            ),
        }
        agent.trials_by_depth[0] = ["t1"]
        agent.frontier[0] = ["t1"]

        # Make llm raise to trigger fallback
        def raising_llm(prompt: str) -> str:
            raise Exception("LLM error")

        agent.llm = raising_llm
        state = MagicMock()
        state.capacity_free.return_value = 1
        state.running_trials.return_value = []

        actions = await agent.decide(state)

        # Should still produce actions via fallback
        assert len(actions) > 0
        assert all(isinstance(a, StartTrial) for a in actions)


class TestParameterValidation:
    """Test parameter validation and adjustment."""

    def test_validate_params_in_bounds(self, agent):
        """Test that parameters are validated to be within bounds."""
        params = {
            "lr": 0.2,  # Above max
            "batch_size": 256,  # Not in choices
            "depth": 10,  # Above max
        }

        validated = agent._validate_params(params)

        # lr should be clamped to max
        assert validated["lr"] <= 0.1

        # batch_size should be from valid choices
        assert validated["batch_size"] in [16, 32, 64, 128]

        # depth should be clamped to max
        assert validated["depth"] <= 5

    def test_validate_params_missing(self, agent):
        """Test that missing parameters are sampled."""
        params = {"lr": 0.05}  # Missing batch_size and depth

        validated = agent._validate_params(params)

        # Should have all parameters
        assert "lr" in validated
        assert "batch_size" in validated
        assert "depth" in validated

        # Values should be valid
        assert 0.001 <= validated["lr"] <= 0.1
        assert validated["batch_size"] in [16, 32, 64, 128]
        assert 1 <= validated["depth"] <= 5


class TestPruning:
    """Test pruning functionality."""

    @pytest.mark.asyncio
    async def test_pruning_weak_branches(self, agent):
        """Test that weak branches are pruned."""
        agent.enable_pruning = True

        # Setup trial tree with strong and weak branches
        agent.trial_tree = {
            "t1": TrialNode(
                trial_id="t1",
                params={"lr": 0.01},
                score=0.9,
                metrics={},
                status="COMPLETED",
                depth=0,
            ),
            "t2": TrialNode(
                trial_id="t2",
                params={"lr": 0.02},
                score=0.5,
                metrics={},
                status="COMPLETED",
                depth=0,
            ),
            "t3": TrialNode(
                trial_id="t3",
                params={"lr": 0.015},
                score=None,
                metrics={},
                status="RUNNING",
                depth=2,
                parent_id="t1",
            ),
            "t4": TrialNode(
                trial_id="t4",
                params={"lr": 0.025},
                score=None,
                metrics={},
                status="RUNNING",
                depth=2,
                parent_id="t2",
            ),
        }

        agent.frontier = {
            0: ["t1"],
            1: [],
        }  # t1 is in frontier, t2 is not, 2 levels exist

        # Mock state with running trials
        state = MagicMock()
        running_views = [
            MagicMock(trial_id="t3", depth=2),
            MagicMock(trial_id="t4", depth=2),
        ]
        state.running_trials.return_value = running_views

        # Get pruning actions
        from hyperion.framework.policy import KillTrial

        actions = agent._get_pruning_actions(state)

        # Should prune t4 (child of weak parent)
        kill_actions = [a for a in actions if isinstance(a, KillTrial)]
        assert len(kill_actions) > 0
        # Note: Exact pruning logic may vary based on implementation


class TestBranchingHistory:
    """Test decision history tracking."""

    def test_branching_history_tracking(self, agent):
        """Test that branching decisions are tracked."""
        decision = BranchingDecision(
            parent_trial_id="t1",
            branch_count=2,
            parameter_focus=["lr"],
            variations=[{"lr": 0.008}, {"lr": 0.012}],
            rationale="Testing lr variations",
            confidence="high",
        )

        agent.branching_history.append(decision)

        assert len(agent.branching_history) == 1
        assert agent.branching_history[0].parent_trial_id == "t1"

    @pytest.mark.asyncio
    async def test_rationale_generation(self, agent):
        """Test rationale generation."""
        agent.last_rationale = "Testing branching strategy"
        agent.decisions_made = 5
        agent.trial_tree = {"t1": MagicMock(), "t2": MagicMock()}

        rationale = await agent.rationale()

        assert "Testing branching strategy" in rationale
        assert "Decisions: 5" in rationale
        assert "Trials: 2" in rationale
