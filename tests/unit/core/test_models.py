"""Tests for core data models."""

from datetime import UTC, datetime

from hyperion.core.models import (
    Experiment,
    ExperimentStatus,
    ObjectiveResult,
    Trial,
    TrialStatus,
)


def test_objective_result_stores_score_and_metrics():
    """ObjectiveResult should store score and optional metrics/artifacts."""
    result = ObjectiveResult(
        score=0.95,
        metrics={"accuracy": 0.95, "loss": 0.05},
        artifacts={"model": "/path/to/model.pkl"},
    )

    assert result.score == 0.95
    assert result.metrics["accuracy"] == 0.95
    assert result.artifacts["model"] == "/path/to/model.pkl"


def test_objective_result_minimal():
    """ObjectiveResult should work with just a score."""
    result = ObjectiveResult(score=0.75)

    assert result.score == 0.75
    assert result.metrics == {}
    assert result.artifacts == {}


def test_trial_initial_state():
    """Trial should start in PENDING state with no score."""
    trial = Trial(
        id="trial-123", experiment_id="exp-456", params={"lr": 0.01, "batch_size": 32}
    )

    assert trial.id == "trial-123"
    assert trial.experiment_id == "exp-456"
    assert trial.params["lr"] == 0.01
    assert trial.status == TrialStatus.PENDING
    assert trial.score is None
    assert trial.started_at is None
    assert trial.ended_at is None


def test_trial_lineage_fields():
    """Trial should support lineage tracking for branching strategies."""
    trial = Trial(
        id="child-trial",
        experiment_id="exp-1",
        params={},
        parent_trial_id="parent-trial",
        depth=2,
        branch_id="branch-main",
        mutation_op="perturb_lr",
    )

    assert trial.parent_trial_id == "parent-trial"
    assert trial.depth == 2
    assert trial.branch_id == "branch-main"
    assert trial.mutation_op == "perturb_lr"


def test_trial_default_lineage():
    """Trial without parent should have depth=0 and no lineage info."""
    trial = Trial(id="root-trial", experiment_id="exp-1", params={})

    assert trial.parent_trial_id is None
    assert trial.depth == 0
    assert trial.branch_id is None
    assert trial.mutation_op is None


def test_trial_state_transitions():
    """Test valid trial state transitions."""
    trial = Trial(id="t1", experiment_id="e1", params={})

    # Valid transitions
    assert trial.status == TrialStatus.PENDING

    # Can transition to RUNNING
    trial.status = TrialStatus.RUNNING
    assert trial.status == TrialStatus.RUNNING

    # Can transition from RUNNING to terminal states
    trial.status = TrialStatus.COMPLETED
    assert trial.status == TrialStatus.COMPLETED


def test_experiment_initial_state():
    """Experiment should start in PENDING state."""
    exp = Experiment(id="exp-789", name="test-experiment", created_at=datetime.now(UTC))

    assert exp.id == "exp-789"
    assert exp.name == "test-experiment"
    assert exp.status == ExperimentStatus.PENDING
    assert exp.config == {}
    assert exp.tags == {}


def test_experiment_with_config():
    """Experiment can store configuration and tags."""
    exp = Experiment(
        id="exp-1",
        name="beam-search-test",
        created_at=datetime.now(UTC),
        config={
            "search_space": {"lr": {"low": 0.001, "high": 0.1}},
            "budget": {"max_trials": 100},
        },
        tags={"team": "research", "priority": "high"},
    )

    assert exp.config["budget"]["max_trials"] == 100
    assert exp.tags["team"] == "research"
